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

/* ============================================================================================ front end
 * Two pieces of ClayDetector that ran through OpenCV / NumPy on the whole frame (Pi 3B, range day 2026-11-19, a
 * tree line: 22 ms a frame of which these were 4.2 ms and, every noise_every-th frame, another 10):
 *
 *   tbot_front_label   the thresholded image -> components (label image, bounding boxes, areas, centroids),
 *                      as cv2.boundingRect + cv2.connectedComponentsWithStats(connectivity=8) on the bounding
 *                      rectangle. The image is nearly all zeros, so it is read eight bytes at a time and only
 *                      the runs of set pixels are worked on. Components are numbered the way OpenCV numbers
 *                      them (in raster order of the first 2x2 block they touch, blocks counted from the
 *                      labelled rectangle's corner), so every later choice by index is the same.
 *   tbot_front_alone   the "stands alone" test of a cluttered frame (set pixels of other blobs in the ring
 *                      around a candidate's box), from the runs instead of an integral image.
 *   tbot_noise_sigma   detector._tiled_sigma on the subsampled residual (gray - bg, or an int16 contrast).
 *
 * The Python / OpenCV code remains the reference (tests compare the two) and runs when the library is not
 * built or a capacity here is exceeded. */
#define TBOT_FRONT_ABI 1

typedef struct {
    int32_t h, w;
    int32_t cap_runs, cap_comps;
    int32_t *runs;       /* cap_runs x 4: y, x0, x1 (exclusive), label; rows ascending, x ascending in a row */
    int32_t *row_start;  /* h + 1: the first run of each row */
    int32_t *labels;     /* h x w, zero outside the runs of the previous call (which are cleared here) */
    int32_t *stats;      /* cap_comps x 5: left, top, width, height, area (row 0, the background, is zeros) */
    double *cents;       /* cap_comps x 2 */
    int32_t n_runs;      /* in/out: the runs now in `labels` */
    int32_t roi[4];      /* out: x, y, w, h of the set pixels (cv2.boundingRect) */
} tbot_front;

TBOT_API int tbot_front_abi(void) { return TBOT_FRONT_ABI; }
TBOT_API int tbot_front_size(void) { return (int)sizeof(tbot_front); }

static inline int32_t front_find(int32_t *parent, int32_t i) {
    while (parent[i] != i) {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    return i;
}

typedef struct { int32_t key, root; } front_order;

static int front_order_cmp(const void *a, const void *b) {
    const int32_t ka = ((const front_order *)a)->key, kb = ((const front_order *)b)->key;
    return (ka > kb) - (ka < kb);
}

/* Returns the number of labels including the background (cv2's n), or -1 when a capacity is exceeded or memory
 * is short: `labels` is then all zeros, n_runs 0, and the caller labels with OpenCV. */
TBOT_API int tbot_front_label(tbot_front *f, const uint8_t *b8) {
    const int h = f->h, w = f->w;
    int32_t *runs = f->runs, *labels = f->labels;
    for (int32_t i = 0; i < f->n_runs; i++) {
        const int32_t *r = runs + 4 * (size_t)i;
        memset(labels + (size_t)r[0] * w + r[1], 0, (size_t)(r[2] - r[1]) * sizeof(int32_t));
    }
    f->n_runs = 0;
    f->roi[0] = f->roi[1] = f->roi[2] = f->roi[3] = 0;

    int32_t n = 0, over = 0;
    int minx = w, maxx = -1, miny = h, maxy = -1;
    for (int y = 0; y < h && !over; y++) {
        const uint8_t *row = b8 + (size_t)y * w;
        f->row_start[y] = n;
        int x = 0;
        while (x < w) {
            while (x + 8 <= w) {            /* eight zero bytes at a time */
                uint64_t v;
                memcpy(&v, row + x, 8);
                if (v) break;
                x += 8;
            }
            while (x < w && !row[x]) x++;
            if (x >= w) break;
            const int x0 = x;
            while (x < w && row[x]) x++;
            if (n >= f->cap_runs) { over = 1; break; }
            int32_t *r = runs + 4 * (size_t)n++;
            r[0] = y; r[1] = x0; r[2] = x; r[3] = 0;
            if (x0 < minx) minx = x0;
            if (x - 1 > maxx) maxx = x - 1;
            if (y < miny) miny = y;
            maxy = y;
        }
    }
    if (over) return -1;
    for (int y = (maxy < 0 ? 0 : maxy + 1); y <= h; y++) f->row_start[y] = n;
    if (n == 0) return 1;

    int32_t *parent = (int32_t *)malloc((size_t)n * sizeof(int32_t));
    if (!parent) return -1;
    for (int32_t i = 0; i < n; i++) parent[i] = i;
    for (int y = miny + 1; y <= maxy; y++) {            /* 8-connectivity: runs of the row above within 1 px */
        const int32_t p1 = f->row_start[y], c1 = f->row_start[y + 1];
        int32_t j = f->row_start[y - 1];
        for (int32_t i = p1; i < c1; i++) {
            const int32_t x0 = runs[4 * (size_t)i + 1], x1 = runs[4 * (size_t)i + 2];
            while (j < p1 && runs[4 * (size_t)j + 2] < x0) j++;
            for (int32_t k = j; k < p1 && runs[4 * (size_t)k + 1] <= x1; k++) {
                const int32_t a = front_find(parent, i), b = front_find(parent, k);
                if (a < b) parent[b] = a;
                else if (b < a) parent[a] = b;
            }
        }
    }

    /* the rectangle OpenCV is given: the frame when the set pixels span more than half of it, else their box */
    const int rw = maxx - minx + 1, rh = maxy - miny + 1;
    const int whole = (int64_t)rw * rh * 2 > (int64_t)h * w;
    const int ox = whole ? 0 : minx, oy = whole ? 0 : miny;
    const int32_t bw2 = (w + 1) / 2;

    /* per root: first 2x2 block (the numbering order), box, area, coordinate sums */
    typedef struct { int32_t key, x0, x1, y0, y1, area, label; int64_t sx, sy; } comp;
    int32_t n_roots = 0;
    for (int32_t i = 0; i < n; i++) n_roots += parent[i] == i;
    if (n_roots + 1 > f->cap_comps) { free(parent); return -1; }
    comp *cs = (comp *)malloc((size_t)n_roots * sizeof(comp));
    front_order *ord = (front_order *)malloc((size_t)n_roots * sizeof(front_order));
    int32_t *slot = (int32_t *)malloc((size_t)n * sizeof(int32_t));
    if (!cs || !ord || !slot) { free(parent); free(cs); free(ord); free(slot); return -1; }
    int32_t nr = 0;
    for (int32_t i = 0; i < n; i++) {
        const int32_t *r = runs + 4 * (size_t)i;
        const int32_t root = front_find(parent, i);
        const int32_t key = ((r[0] - oy) >> 1) * bw2 + ((r[1] - ox) >> 1);
        const int32_t len = r[2] - r[1];
        const int64_t sx = (int64_t)(r[1] - ox + r[2] - 1 - ox) * len / 2;
        if (root == i) {                    /* a root is the first run of its component in raster order */
            slot[i] = nr;
            comp *c = cs + nr++;
            c->key = key; c->x0 = r[1]; c->x1 = r[2]; c->y0 = c->y1 = r[0]; c->area = len;
            c->sx = sx; c->sy = (int64_t)(r[0] - oy) * len;
        } else {
            comp *c = cs + slot[root];
            if (key < c->key) c->key = key;
            if (r[1] < c->x0) c->x0 = r[1];
            if (r[2] > c->x1) c->x1 = r[2];
            c->y1 = r[0];
            c->area += len; c->sx += sx; c->sy += (int64_t)(r[0] - oy) * len;
        }
    }
    for (int32_t k = 0; k < nr; k++) { ord[k].key = cs[k].key; ord[k].root = k; }
    qsort(ord, (size_t)nr, sizeof(front_order), front_order_cmp);
    memset(f->stats, 0, 5 * sizeof(int32_t));
    f->cents[0] = f->cents[1] = 0.0;
    for (int32_t k = 0; k < nr; k++) {
        comp *c = cs + ord[k].root;
        c->label = k + 1;
        int32_t *s = f->stats + 5 * (size_t)(k + 1);
        s[0] = c->x0; s[1] = c->y0; s[2] = c->x1 - c->x0; s[3] = c->y1 - c->y0 + 1; s[4] = c->area;
        /* as OpenCV on the rectangle, then moved by its corner (the same two operations, the same rounding) */
        f->cents[2 * (size_t)(k + 1)] = (double)c->sx / (double)c->area + (double)ox;
        f->cents[2 * (size_t)(k + 1) + 1] = (double)c->sy / (double)c->area + (double)oy;
    }
    for (int32_t i = 0; i < n; i++) {
        int32_t *r = runs + 4 * (size_t)i;
        const int32_t lab = cs[slot[front_find(parent, i)]].label;
        r[3] = lab;
        int32_t *dst = labels + (size_t)r[0] * w + r[1];
        for (int32_t x = r[2] - r[1]; x > 0; x--) *dst++ = lab;
    }
    free(parent); free(cs); free(ord); free(slot);
    f->n_runs = n;
    f->roi[0] = minx; f->roi[1] = miny; f->roi[2] = rw; f->roi[3] = rh;
    return nr + 1;
}

/* alone[i] = 1 when, in the ring of clip(max(w, h), ring_min, ring_max_px) pixels around the box of component
 * idx[i] + 1 (clipped to the rectangle of the set pixels), at most ring_frac of the ring is set pixels that are
 * not its own (detector._measure, clutter_ring_max). */
TBOT_API void tbot_front_alone(const tbot_front *f, const intptr_t *idx, int32_t n, int32_t ring_min,
                               int32_t ring_max_px, double ring_frac, uint8_t *alone) {
    const int rx = f->roi[0], ry = f->roi[1], rw = f->roi[2], rh = f->roi[3];
    for (int32_t i = 0; i < n; i++) {
        const int32_t *s = f->stats + 5 * (size_t)(idx[i] + 1);
        const int bx = s[0] - rx, by = s[1] - ry, bw = s[2], bh = s[3];
        int m = bw > bh ? bw : bh;
        if (m < ring_min) m = ring_min;
        if (m > ring_max_px) m = ring_max_px;
        const int x0 = bx - m > 0 ? bx - m : 0, y0 = by - m > 0 ? by - m : 0;
        const int x1 = bx + bw + m < rw ? bx + bw + m : rw, y1 = by + bh + m < rh ? by + bh + m : rh;
        const int ax0 = x0 + rx, ax1 = x1 + rx;
        int64_t inside = 0;
        for (int y = y0 + ry; y < y1 + ry; y++) {
            const int32_t e = f->row_start[y + 1];
            for (int32_t k = f->row_start[y]; k < e; k++) {
                const int32_t *r = f->runs + 4 * (size_t)k;
                if (r[1] >= ax1) break;
                const int lo = r[1] > ax0 ? r[1] : ax0, hi = r[2] < ax1 ? r[2] : ax1;
                if (hi > lo) inside += hi - lo;
            }
        }
        int64_t ring = (int64_t)(x1 - x0) * (y1 - y0) - (int64_t)bw * bh;
        if (ring < 1) ring = 1;
        alone[i] = (double)(inside - s[4]) <= ring_frac * (double)ring;
    }
}

static double front_erfinv(double y) {      /* detector._erfinv: bisection, the same 40 steps */
    double lo = 0.0, hi = 4.0;
    for (int i = 0; i < 40; i++) {
        const double mid = 0.5 * (lo + hi);
        if (erf(mid) < y) lo = mid;
        else hi = mid;
    }
    return 0.5 * (lo + hi);
}

static int front_double_cmp(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* detector._tiled_sigma of the residual sampled every `step` pixels: gray - bg (uint8 each) or c16 (int16), mask
 * (bool, may be NULL) the pixels that count. NaN when the frame is too small or too masked to tile: the caller
 * then runs the Python estimate. */
TBOT_API double tbot_noise_sigma(const uint8_t *gray, const uint8_t *bg, const int16_t *c16, const uint8_t *mask,
                                 int32_t h, int32_t w, int32_t step, int32_t rows, int32_t cols, int32_t dev_max,
                                 int32_t tile_min, double quantile, double tile_quantile, double sigma_floor) {
    if (step < 1 || rows < 1 || cols < 1 || dev_max < 1 || rows * cols > 256) return NAN;
    const int hs = (h + step - 1) / step, ws = (w + step - 1) / step;
    if (hs < 4 * rows || ws < 4 * cols) return NAN;
    const int n_tiles = rows * cols, width = 2 * dev_max + 1;
    int32_t *hist = (int32_t *)calloc((size_t)n_tiles * width + (size_t)width + 1, sizeof(int32_t));
    double *sig = (double *)malloc(2 * (size_t)n_tiles * sizeof(double));
    if (!hist || !sig) { free(hist); free(sig); return NAN; }
    int32_t *cum0 = hist + (size_t)n_tiles * width;
    double *cnt = sig + n_tiles;
    for (int i = 0; i < hs; i++) {
        int ty = i * rows / hs;
        if (ty > rows - 1) ty = rows - 1;
        const size_t off = (size_t)i * step * w;
        for (int j = 0; j < ws; j++) {
            const size_t at = off + (size_t)j * step;
            if (mask && !mask[at]) continue;
            int tx = j * cols / ws;
            if (tx > cols - 1) tx = cols - 1;
            int v = c16 ? (int)c16[at] : (int)gray[at] - (int)bg[at];
            if (v < -dev_max) v = -dev_max;
            if (v > dev_max) v = dev_max;
            hist[(size_t)(ty * cols + tx) * width + v + dev_max]++;
        }
    }
    int n_ok = 0;
    for (int t = 0; t < n_tiles; t++) {
        const int32_t *ht = hist + (size_t)t * width;
        int64_t total = 0;
        for (int b = 0; b < width; b++) total += ht[b];
        if (total < tile_min) continue;
        const double nn = (double)total;
        cum0[0] = 0;
        int med = -1;
        for (int b = 0; b < width; b++) {
            cum0[b + 1] = cum0[b] + ht[b];
            if (med < 0 && (double)cum0[b + 1] >= 0.5 * nn) med = b;
        }
        if (med < 0) med = 0;
        int kbest = 0;
        double dbest = 0.0, fbest = 0.0;
        for (int k = 0; k <= dev_max; k++) {
            const int hi = (med + k < width - 1 ? med + k : width - 1) + 1, lo = med - k > 0 ? med - k : 0;
            const double frac = (double)(cum0[hi] - cum0[lo]) / nn;
            const double d = fabs(frac - quantile);
            if (k == 0 || d < dbest) { dbest = d; kbest = k; fbest = frac; }
        }
        if (fbest < 0.05) fbest = 0.05;
        if (fbest > 0.98) fbest = 0.98;
        sig[n_ok] = ((double)kbest + 0.5) / (sqrt(2.0) * front_erfinv(fbest));
        cnt[n_ok++] = nn;
    }
    const int need = n_tiles / 4 > 2 ? n_tiles / 4 : 2;
    if (n_ok < need) { free(hist); free(sig); return NAN; }
    qsort(sig, (size_t)n_ok, sizeof(double), front_double_cmp);
    qsort(cnt, (size_t)n_ok, sizeof(double), front_double_cmp);
    int at = (int)(tile_quantile * (double)n_ok);
    if (at > n_ok - 1) at = n_ok - 1;
    double q = sig[at];
    const double med_n = (n_ok & 1) ? cnt[n_ok / 2] : (cnt[n_ok / 2 - 1] + cnt[n_ok / 2]) / 2.0;
    q *= 1.0 + (0.674 * 1.3) / sqrt(med_n);
    free(hist); free(sig);
    return q > sigma_floor ? q : sigma_floor;
}

/* One s x s box mean with the rounding of this machine's cv2.resize(INTER_AREA): OpenCV's own loop rounds
 * sum / s^2 half to even (through a float), its ARM back end half up. detector._area_rounding() finds out
 * which once and passes it here. */
static inline uint8_t front_box_round(uint32_t sum, int area, float scale, int shift, int half_up) {
    if (shift >= 1) {                     /* a power of two: in integers (no call per pixel) */
        const uint32_t half = (uint32_t)1 << (shift - 1), q = sum >> shift, r = sum & (2 * half - 1);
        return (uint8_t)(q + (half_up ? r >= half : (r > half) | ((r == half) & q & 1)));
    }
    if (half_up) return (uint8_t)((2 * sum + (uint32_t)area) / (2 * (uint32_t)area));
    return (uint8_t)lrintf((float)sum * scale);
}

static int front_shift_of(int area) {
    for (int k = 0; k < 16; k++)
        if (area == (1 << k)) return k;
    return -1;
}

/* dst (h/s x w/s) = src (h x w) averaged over s x s boxes: cv2.resize(src, (w/s, h/s), INTER_AREA) for h and w
 * multiples of s (2 <= s <= 16). Returns 0, or -1 when the sizes are not that. */
TBOT_API int tbot_front_box(const uint8_t *src, int32_t h, int32_t w, int32_t s, int32_t half_up, uint8_t *dst) {
    if (s < 2 || s > 16 || h % s || w % s) return -1;
    const int dh = h / s, dw = w / s, area = s * s, shift = front_shift_of(area);
    const float scale = 1.f / (float)area;
    uint16_t *col = (uint16_t *)malloc((size_t)w * sizeof(uint16_t));
    if (!col) return -1;
    for (int y = 0; y < dh; y++) {
        const uint8_t *restrict r = src + (size_t)y * s * w;
        uint16_t *restrict cc = col;
        for (int x = 0; x < w; x++) cc[x] = r[x];
        for (int k = 1; k < s; k++) {
            r += w;
            for (int x = 0; x < w; x++) cc[x] = (uint16_t)(cc[x] + r[x]);
        }
        uint8_t *d = dst + (size_t)y * dw;
        const uint16_t *c = col;
        for (int x = 0; x < dw; x++, c += s) {
            uint32_t sum = 0;
            for (int k = 0; k < s; k++) sum += c[k];
            d[x] = front_box_round(sum, area, scale, shift, half_up);
        }
    }
    free(col);
    return 0;
}

/* The block statistics of ClayDetector._local_threshold in one pass: dst (h/cell x w/cell) = the mean over each
 * cell of min(255, gain * absd) at every other pixel of every other row, i.e. cv2.resize(NEAREST, half size),
 * cv2.convertScaleAbs(alpha = gain), cv2.resize(INTER_AREA). cell even, h and w multiples of it. */
TBOT_API int tbot_front_cells(const uint8_t *absd, int32_t h, int32_t w, int32_t cell, int32_t gain,
                              int32_t half_up, uint8_t *dst) {
    if (cell < 4 || cell > 32 || (cell & 1) || h % cell || w % cell || gain < 1 || gain > 255) return -1;
    const int k = cell / 2, ch = h / cell, cw = w / cell, area = k * k, shift = front_shift_of(area);
    const float scale = 1.f / (float)area;
    const int half_w = w / 2;
    uint16_t *col = (uint16_t *)malloc((size_t)half_w * sizeof(uint16_t));
    if (!col) return -1;
    for (int y = 0; y < ch; y++) {
        memset(col, 0, (size_t)half_w * sizeof(uint16_t));
        for (int j = 0; j < k; j++) {
            const uint8_t *restrict r = absd + (size_t)(y * cell + 2 * j) * w;
            uint16_t *restrict cc = col;
            for (int i = 0; i < half_w; i++) {          /* min(255, gain * v) of every other pixel */
                const unsigned v = (unsigned)r[2 * i] * (unsigned)gain;
                cc[i] = (uint16_t)(cc[i] + (v > 255u ? 255u : v));
            }
        }
        uint8_t *d = dst + (size_t)y * cw;
        const uint16_t *c = col;
        for (int x = 0; x < cw; x++, c += k) {
            uint32_t sum = 0;
            for (int i = 0; i < k; i++) sum += c[i];
            d[x] = front_box_round(sum, area, scale, shift, half_up);
        }
    }
    free(col);
    return 0;
}

/* ================================================================================================ motion
 * The per-frame work of the motion-first detector (tbot/vision/motion.py, MotionDetector.detect), which stays as
 * the reference and the fallback (vision.native = false, TBOT_NATIVE=0, or this family missing / stale).
 *
 *   tbot_motion_prep     box_down2 chain frame -> working -> fine -> coarse, texture_map and texture_tensor
 *   tbot_motion_reg1     _register up to the acceptance test: coarse warp, ratio-of-means gain, coarse SAD
 *                        surface, fine search with recentring, parabolic sub-pixel, texture counts
 *   tbot_motion_reg2     _register after it: warp by the final transform, photometric_fit, _local_field
 *   tbot_motion_offsets  _field_extra + block_offsets for the working-resolution warp
 *   tbot_motion_kernel   motion_kernel: warps, gains, envelope residual, cell sums, noise sigma, CFAR, compare
 *   tbot_motion_measure  bounding box, 3x3 closing, 8-connected labels (OpenCV's numbering), moments, ranking and
 *                        the per-candidate numbers of MotionDetector._measure
 * and the primitives they are made of (exported for the parity tests: tests/test_native_motion.py).
 *
 * Integer functions are exact. Floating-point code follows the NumPy expressions operation by operation (this
 * file is built with -ffp-contract=off): sums of float64 arrays are NumPy's pairwise summation, percentiles its
 * "linear" method, medians the mean of the two middle values, max / min where Python's builtins are used have
 * their argument-order semantics. Anything computed with BLAS / LAPACK / cv2.findHomography in the reference
 * (the prior, the level matrices) stays in Python and is passed in. */
#define TBOT_MOTION_ABI 5
#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif
#define MO_DET_COLS 15

static inline int mo_imin(int a, int b) { return a < b ? a : b; }
static inline int mo_imax(int a, int b) { return a > b ? a : b; }
static inline double mo_pymax(double a, double b) { return b > a ? b : a; }  /* Python max(a, b) */
static inline double mo_pymin(double a, double b) { return b < a ? b : a; }  /* Python min(a, b) */

/* NumPy's add.reduce of a contiguous float64 run: identity 0.0 + pairwise sum (blocks of 128, 8 accumulators) */
static double mo_pairwise(const double *a, intptr_t n) {
    if (n < 8) {
        double res = -0.0;
        for (intptr_t i = 0; i < n; i++) res += a[i];
        return res;
    }
    if (n <= 128) {
        double r[8];
        intptr_t i;
        for (int k = 0; k < 8; k++) r[k] = a[k];
        for (i = 8; i < n - (n % 8); i += 8)
            for (int k = 0; k < 8; k++) r[k] += a[i + k];
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; i++) res += a[i];
        return res;
    }
    intptr_t n2 = n / 2;
    n2 -= n2 % 8;
    return mo_pairwise(a, n2) + mo_pairwise(a + n2, n - n2);
}

/* add.reduce of a contiguous float64 array: the identity 0.0, plus the pairwise sum of each run of `chunk`
 * elements (NumPy before 2.3 hands a reduction to its inner loop in buffer-sized runs of 8192; chunk <= 0: one
 * run). _native.py finds out which this NumPy does. */
static double mo_sum(const double *a, intptr_t n, int32_t chunk) {
    if (chunk <= 0 || n <= chunk) return 0.0 + mo_pairwise(a, n);
    double s = 0.0;
    for (intptr_t i = 0; i < n; i += chunk) s += mo_pairwise(a + i, n - i < chunk ? n - i : chunk);
    return s;
}

static double mo_mean(const double *a, intptr_t n, int32_t chunk) { return mo_sum(a, n, chunk) / (double)n; }

/* ndarray.astype(np.int32) of a float64 as this machine's conversion instruction does it */
static inline int32_t mo_to_i32(double v) {
#if defined(__aarch64__) || defined(__arm__)
    if (v != v) return 0;
    if (v >= 2147483647.0) return INT32_MAX;
    if (v <= -2147483648.0) return INT32_MIN;
    return (int32_t)v;
#else
    if (!(v > -2147483649.0 && v < 2147483648.0)) return INT32_MIN;
    return (int32_t)v;
#endif
}

static void mo_sort_doubles(double *a, int n) {   /* insertion sort (small arrays) */
    for (int i = 1; i < n; i++) {
        const double v = a[i];
        int j = i - 1;
        while (j >= 0 && a[j] > v) { a[j + 1] = a[j]; j--; }
        a[j + 1] = v;
    }
}

/* np.median of n finite values (destroys a): the middle one, or (0.0 + (lo + hi)) / 2 */
static double mo_median(double *a, int n) {
    mo_sort_doubles(a, n);
    if (n & 1) return (0.0 + a[n / 2]) / 1.0;
    return (0.0 + (a[n / 2 - 1] + a[n / 2])) / 2.0;
}

/* the k-th smallest of n doubles (np.partition(a, k)[k]); destroys a */
static double mo_select(double *a, int n, int k) {
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        const double pv = a[lo + (hi - lo) / 2];
        int i = lo, j = hi;
        while (i <= j) {
            while (a[i] < pv) i++;
            while (a[j] > pv) j--;
            if (i <= j) { const double t = a[i]; a[i] = a[j]; a[j] = t; i++; j--; }
        }
        if (k <= j) hi = j;
        else if (k >= i) lo = i;
        else return a[k];
    }
    return a[k];
}

/* ------------------------------------------------------------------------------------- primitives */
TBOT_API int tbot_motion_abi(void) { return TBOT_MOTION_ABI; }

TBOT_API double tbot_motion_sum(const double *a, int64_t n, int32_t chunk) { return mo_sum(a, (intptr_t)n, chunk); }

/* box_down2: dst (h/2 x w/2) = (s00 + s01 + s10 + s11 + 2) >> 2 */
TBOT_API void tbot_motion_box_down2(const uint8_t *src, int32_t h, int32_t w, uint8_t *dst) {
    const int dh = h / 2, dw = w / 2;
    for (int y = 0; y < dh; y++) {
        const uint8_t *restrict r0 = src + (size_t)(2 * y) * w, *restrict r1 = r0 + w;
        uint8_t *restrict d = dst + (size_t)y * dw;
        int x = 0;
#if defined(__ARM_NEON)
        for (; x + 16 <= dw; x += 16) {          /* pairwise sums of both rows, (s + 2) >> 2 narrowed */
            uint16x8_t s0 = vpadalq_u8(vpaddlq_u8(vld1q_u8(r0 + 2 * x)), vld1q_u8(r1 + 2 * x));
            uint16x8_t s1 = vpadalq_u8(vpaddlq_u8(vld1q_u8(r0 + 2 * x + 16)), vld1q_u8(r1 + 2 * x + 16));
            vst1q_u8(d + x, vcombine_u8(vrshrn_n_u16(s0, 2), vrshrn_n_u16(s1, 2)));
        }
#endif
        for (; x < dw; x++)
            d[x] = (uint8_t)(((unsigned)r0[2 * x] + r0[2 * x + 1] + r1[2 * x] + r1[2 * x + 1] + 2u) >> 2);
    }
}

/* texture_map (dst 0 / 255; last row and column 0) and, when valid is given, texture_tensor over tex & valid */
TBOT_API void tbot_motion_texture(const uint8_t *img, int32_t h, int32_t w, int32_t thr, uint8_t *dst,
                                  const uint8_t *valid, int64_t *tensor) {
    int64_t sxx = 0, syy = 0, sxy = 0;
    for (int y = 0; y < h; y++) {
        uint8_t *restrict d = dst + (size_t)y * w;
        if (y == h - 1) { memset(d, 0, (size_t)w); break; }
        const uint8_t *restrict a = img + (size_t)y * w, *restrict b = a + w;
        for (int x = 0; x < w - 1; x++) {
            const int gx = (int)a[x + 1] - (int)a[x], gy = (int)b[x] - (int)a[x];
            const int ax = gx < 0 ? -gx : gx, ay = gy < 0 ? -gy : gy;
            d[x] = (ax > ay ? ax : ay) > thr ? 255 : 0;
        }
        if (w > 0) d[w - 1] = 0;
        if (valid) {
            const uint8_t *restrict v = valid + (size_t)y * w;
            int64_t rxx = 0, ryy = 0, rxy = 0;
            for (int x = 0; x < w - 1; x++) {
                if (!d[x] || !v[x]) continue;
                const int gx = (int)a[x + 1] - (int)a[x], gy = (int)b[x] - (int)a[x];
                rxx += gx * gx; ryy += gy * gy; rxy += gx * gy;
            }
            sxx += rxx; syy += ryy; sxy += rxy;
        }
    }
    if (tensor) { tensor[0] = sxx; tensor[1] = syy; tensor[2] = sxy; }
}

/* the whole prep of detect(): frame (H x W) -> working (steps box_down2, or a copy) -> fine (n_fine more, or a copy)
 * -> coarse, texture map of fine and its tensor over curv. scratch: 2 x (H/2) x (W/2) bytes. */
TBOT_API void tbot_motion_prep(const uint8_t *img, int32_t H, int32_t W, int32_t steps, int32_t n_fine,
                               uint8_t *scratch, uint8_t *work, uint8_t *fine, uint8_t *coarse, uint8_t *tex,
                               int32_t tex_thr, const uint8_t *curv, int64_t *tensor) {
    const size_t half = (size_t)(H / 2) * (size_t)(W / 2);
    const uint8_t *src = img;
    int h = H, w = W;
    if (steps == 0) memcpy(work, img, (size_t)H * W);
    for (int k = 0; k < steps; k++) {
        uint8_t *dst = k == steps - 1 ? work : scratch + (k & 1) * half;
        tbot_motion_box_down2(src, h, w, dst);
        src = dst; h /= 2; w /= 2;
    }
    src = work;
    if (n_fine == 0) memcpy(fine, work, (size_t)h * w);
    for (int k = 0; k < n_fine; k++) {
        uint8_t *dst = k == n_fine - 1 ? fine : scratch + (k & 1) * half;
        tbot_motion_box_down2(src, h, w, dst);
        src = dst; h /= 2; w /= 2;
    }
    tbot_motion_box_down2(fine, h, w, coarse);
    tbot_motion_texture(fine, h, w, tex_thr, tex, curv, tensor);
}

/* block_offsets: m row-major (rows = 2 or 3) destination -> source; ex_add / ey_add (h x nb) or NULL; rx, ry (h x
 * nb) receive the rounding residuals; res (2) their means (may be NULL). */
static void mo_offsets(const double *m, int rows, int h, int w, int block, const double *ex_add,
                       const double *ey_add, int32_t *dx, int32_t *dy, double *rx, double *ry, double *res,
                       int32_t chunk) {
    const int nb = (w + block - 1) / block;
    for (int y = 0; y < h; y++) {
        const double yd = (double)y;
        for (int b = 0; b < nb; b++) {
            const int x0 = b * block, x1 = mo_imin(x0 + block, w);
            const double xc = (double)(x0 + x1 - 1) * 0.5;
            const double den = rows == 3 ? m[6] * xc + m[7] * yd + m[8] : 1.0;
            double ex = (m[0] * xc + m[1] * yd + m[2]) / den - xc;
            double ey = (m[3] * xc + m[4] * yd + m[5]) / den - yd;
            const size_t k = (size_t)y * nb + b;
            if (ex_add) { ex = ex + ex_add[k]; ey = ey + ey_add[k]; }
            const double fx = floor(ex + 0.5), fy = floor(ey + 0.5);
            dx[k] = mo_to_i32(fx);
            dy[k] = mo_to_i32(fy);
            rx[k] = ex - fx;
            ry[k] = ey - fy;
        }
    }
    if (res) {
        res[0] = mo_mean(rx, (intptr_t)h * nb, chunk);
        res[1] = mo_mean(ry, (intptr_t)h * nb, chunk);
    }
}

TBOT_API void tbot_motion_block_offsets(const double *m, int32_t rows, int32_t h, int32_t w, int32_t block,
                                        const double *ex_add, const double *ey_add, int32_t *dx, int32_t *dy,
                                        double *rx, double *ry, double *res, int32_t chunk) {
    mo_offsets(m, rows, h, w, block, ex_add, ey_add, dx, dy, rx, ry, res, chunk);
}

/* warp_gather: cv2.remap INTER_NEAREST of an int16 map (x + (int16)dx, y + (int16)dy, int16 arithmetic),
 * BORDER_CONSTANT 0, for the image and its validity */
static inline void mo_zero2(uint8_t *d, uint8_t *e, int x0, int x1) {
    if (x1 > x0) { memset(d + x0, 0, (size_t)(x1 - x0)); memset(e + x0, 0, (size_t)(x1 - x0)); }
}

TBOT_API void tbot_motion_warp(const uint8_t *src, const uint8_t *sv, int32_t h, int32_t w, const int32_t *dx,
                               const int32_t *dy, int32_t block, uint8_t *dst, uint8_t *dv) {
    const int nb = (w + block - 1) / block;
    for (int y = 0; y < h; y++) {
        uint8_t *restrict d = dst + (size_t)y * w, *restrict e = dv + (size_t)y * w;
        const int32_t *rdx = dx + (size_t)y * nb, *rdy = dy + (size_t)y * nb;
        int b = 0;
        while (b < nb) {
            const int ox = (int16_t)rdx[b], oy = (int16_t)rdy[b];
            int b1 = b + 1;                    /* blocks with the same offsets: one copy */
            while (b1 < nb && (int16_t)rdx[b1] == ox && (int16_t)rdy[b1] == oy) b1++;
            const int x0 = b * block, x1 = mo_imin(b1 * block, w);
            b = b1;
            const int sy = (int16_t)(y + oy);
            if (sy < 0 || sy >= h) { mo_zero2(d, e, x0, x1); continue; }
            const uint8_t *s = src + (size_t)sy * w, *t = sv + (size_t)sy * w;
            if (x0 + ox >= -32768 && x1 - 1 + ox <= 32767) {      /* no int16 wrap in this run */
                const int a = mo_imax(x0, -ox), c = mo_imin(x1, w - ox);
                if (c <= a) { mo_zero2(d, e, x0, x1); continue; }
                mo_zero2(d, e, x0, a);
                memcpy(d + a, s + a + ox, (size_t)(c - a));
                memcpy(e + a, t + a + ox, (size_t)(c - a));
                mo_zero2(d, e, c, x1);
            } else {
                for (int x = x0; x < x1; x++) {
                    const int sx = (int16_t)(x + ox);
                    if (sx >= 0 && sx < w) { d[x] = s[sx]; e[x] = t[sx]; }
                    else { d[x] = 0; e[x] = 0; }
                }
            }
        }
    }
}

static void mo_gain_lut(int64_t g, int64_t o, uint8_t *lut) {
    for (int v = 0; v < 256; v++) {
        int64_t t = (int64_t)v * g + o;
        t = t >= 0 ? t >> 8 : -((-t + 255) >> 8);        /* floor(t / 256) */
        lut[v] = (uint8_t)(t < 0 ? 0 : t > 255 ? 255 : t);
    }
}

/* apply_gain_q8: dst = clamp(floor((img * g + o) / 256), 0, 255) (in place allowed) */
#if defined(__aarch64__)
static void mo_gain_fast(const uint8_t *img, int64_t n, int64_t g, int64_t o, uint8_t *dst);
#endif

TBOT_API void tbot_motion_gain(const uint8_t *img, int64_t n, int64_t g, int64_t o, uint8_t *dst) {
#if defined(__aarch64__)
    mo_gain_fast(img, n, g, o, dst);
    return;
#endif
    if (g == 256 && o == 128) {
        if (dst != img) memcpy(dst, img, (size_t)n);
        return;
    }
    const int64_t ag = g < 0 ? -g : g, ao = o < 0 ? -o : o;
    if (ag * 255 + ao < ((int64_t)1 << 30)) {      /* int32 arithmetic, no overflow: vectorises (no table) */
        const int32_t g32 = (int32_t)g, o32 = (int32_t)o;
        for (int64_t i = 0; i < n; i++) {
            int32_t t = ((int32_t)img[i] * g32 + o32) >> 8;   /* arithmetic shift: floor */
            t = t < 0 ? 0 : t;
            dst[i] = (uint8_t)(t > 255 ? 255 : t);
        }
        return;
    }
    uint8_t lut[256];
    mo_gain_lut(g, o, lut);
    for (int64_t i = 0; i < n; i++) dst[i] = lut[img[i]];
}

#if defined(__aarch64__)
/* the gain through its 256-byte table: NEON table lookups on aarch64 (exact: the same table) */
static void mo_gain_fast(const uint8_t *img, int64_t n, int64_t g, int64_t o, uint8_t *dst) {
    if (g == 256 && o == 128) {
        if (dst != img) memcpy(dst, img, (size_t)n);
        return;
    }
    uint8_t lut[256];
    mo_gain_lut(g, o, lut);
    int64_t i = 0;
#if defined(__aarch64__)
    const uint8x16x4_t t0 = vld1q_u8_x4(lut), t1 = vld1q_u8_x4(lut + 64), t2 = vld1q_u8_x4(lut + 128),
                       t3 = vld1q_u8_x4(lut + 192);
    const uint8x16_t k64 = vdupq_n_u8(64);
    for (; i + 16 <= n; i += 16) {
        const uint8x16_t v = vld1q_u8(img + i);
        uint8x16_t r = vqtbl4q_u8(t0, v);
        const uint8x16_t v1 = vsubq_u8(v, k64), v2 = vsubq_u8(v1, k64), v3 = vsubq_u8(v2, k64);
        r = vqtbx4q_u8(r, t1, v1);
        r = vqtbx4q_u8(r, t2, v2);
        r = vqtbx4q_u8(r, t3, v3);
        vst1q_u8(dst + i, r);
    }
#endif
    for (; i < n; i++) dst[i] = lut[img[i]];
}
#endif

/* env x env max (or min) inside the frame (cv2.dilate / cv2.erode with a rectangle, anchor at its centre) */
static void mo_morph(const uint8_t *src, int h, int w, int k, int is_max, uint8_t *tmp, uint8_t *dst) {
    const int a = k / 2, bb = k - 1 - a;
    for (int y = 0; y < h; y++) {                         /* rows */
        const uint8_t *restrict s = src + (size_t)y * w;
        uint8_t *restrict t = tmp + (size_t)y * w;
        memcpy(t, s, (size_t)w);
        for (int o = -a; o <= bb; o++) {
            if (o == 0) continue;
            const int x0 = mo_imax(0, -o), x1 = mo_imin(w, w - o);
            if (is_max) for (int x = x0; x < x1; x++) t[x] = s[x + o] > t[x] ? s[x + o] : t[x];
            else for (int x = x0; x < x1; x++) t[x] = s[x + o] < t[x] ? s[x + o] : t[x];
        }
    }
    for (int y = 0; y < h; y++) {                         /* columns */
        uint8_t *restrict d = dst + (size_t)y * w;
        memcpy(d, tmp + (size_t)y * w, (size_t)w);
        for (int o = -a; o <= bb; o++) {
            if (o == 0 || y + o < 0 || y + o >= h) continue;
            const uint8_t *restrict t = tmp + (size_t)(y + o) * w;
            if (is_max) for (int x = 0; x < w; x++) d[x] = t[x] > d[x] ? t[x] : d[x];
            else for (int x = 0; x < w; x++) d[x] = t[x] < d[x] ? t[x] : d[x];
        }
    }
}

/* envelope_residual. scratch: 5 x h x w bytes. mask may be NULL. */
TBOT_API void tbot_motion_envelope(const uint8_t *cur, const uint8_t *wa, const uint8_t *wb, const uint8_t *va,
                                   const uint8_t *vb, const uint8_t *mask, int32_t h, int32_t w, int32_t env,
                                   uint8_t *scratch, uint8_t *D) {
    const size_t n = (size_t)h * w;
    uint8_t *hiA = scratch, *loA = scratch + n, *hiB = scratch + 2 * n, *loB = scratch + 3 * n, *tmp = scratch + 4 * n;
    mo_morph(wa, h, w, env, 1, tmp, hiA);
    mo_morph(wa, h, w, env, 0, tmp, loA);
    mo_morph(wb, h, w, env, 1, tmp, hiB);
    mo_morph(wb, h, w, env, 0, tmp, loB);
    const uint8_t *restrict c = cur;
    uint8_t *restrict out = D;
    for (size_t i = 0; i < n; i++) {
        const uint8_t x = c[i];
        const uint8_t ba = x > hiA[i] ? (uint8_t)(x - hiA[i]) : 0, bbv = x > hiB[i] ? (uint8_t)(x - hiB[i]) : 0;
        const uint8_t da = loA[i] > x ? (uint8_t)(loA[i] - x) : 0, db = loB[i] > x ? (uint8_t)(loB[i] - x) : 0;
        const uint8_t br = ba < bbv ? ba : bbv, dk = da < db ? da : db;
        out[i] = (uint8_t)((br > dk ? br : dk) & va[i] & vb[i] & (mask ? mask[i] : 0xFF));
    }
}

/* cell_sums: rows y % row_step == 0, (ceil(h / cell) x ceil(w / cell)) int64 */
TBOT_API void tbot_motion_cell_sums(const uint8_t *img, int32_t h, int32_t w, int32_t cell, int32_t row_step,
                                    int64_t *out) {
    const int ch = (h + cell - 1) / cell, cw = (w + cell - 1) / cell;
    memset(out, 0, (size_t)ch * cw * sizeof(int64_t));
    for (int y = 0; y < h; y += row_step) {
        const uint8_t *restrict r = img + (size_t)y * w;
        int64_t *o = out + (size_t)(y / cell) * cw;
        for (int j = 0; j < cw; j++) {
            const int x0 = j * cell, x1 = mo_imin(x0 + cell, w);
            uint32_t s = 0;
            for (int x = x0; x < x1; x++) s += r[x];
            o[j] += s;
        }
    }
}

/* cell_absdiff_sums: per cell over rows y % row_step == 0, sum of |cur - wa| & valid and count of valid != 0 */
TBOT_API void tbot_motion_absdiff_sums(const uint8_t *cur, const uint8_t *wa, const uint8_t *valid, int32_t h,
                                       int32_t w, int32_t cell, int32_t row_step, int64_t *sums, int64_t *counts) {
    const int ch = (h + cell - 1) / cell, cw = (w + cell - 1) / cell;
    memset(sums, 0, (size_t)ch * cw * sizeof(int64_t));
    memset(counts, 0, (size_t)ch * cw * sizeof(int64_t));
    for (int y = 0; y < h; y += row_step) {
        const uint8_t *restrict a = cur + (size_t)y * w, *restrict b = wa + (size_t)y * w, *restrict v = valid + (size_t)y * w;
        int64_t *so = sums + (size_t)(y / cell) * cw, *co = counts + (size_t)(y / cell) * cw;
        for (int j = 0; j < cw; j++) {
            const int x0 = j * cell, x1 = mo_imin(x0 + cell, w);
            uint32_t s = 0, c = 0;
            for (int x = x0; x < x1; x++) {
                const uint8_t d = a[x] > b[x] ? (uint8_t)(a[x] - b[x]) : (uint8_t)(b[x] - a[x]);
                s += (uint8_t)(d & v[x]);
                c += v[x] != 0;
            }
            so[j] += s;
            co[j] += c;
        }
    }
}

/* noise_sigma_from_cells; NaN when no cell qualifies */
TBOT_API double tbot_motion_noise(const int64_t *sums, const int64_t *counts, int32_t n, int32_t cell,
                                  int32_t row_step, double quantile, double min_fill, double abs_to_sigma) {
    const int64_t full = (int64_t)cell * ((cell + row_step - 1) / row_step);
    double *means = (double *)malloc((size_t)(n > 0 ? n : 1) * sizeof(double));
    if (!means) return NAN;
    int m = 0;
    for (int i = 0; i < n; i++)
        if ((double)counts[i] >= min_fill * (double)full) means[m++] = (double)sums[i] / (double)counts[i];
    if (m == 0) { free(means); return NAN; }
    const int k = (int)(quantile * (double)(m - 1));
    const double v = mo_select(means, m, k);
    free(means);
    return v * abs_to_sigma;
}

/* k x k box sums clipped to the array, through an integral image (int64: exact) */
static void mo_box_clipped(const int64_t *a, int h, int w, int k, int64_t *ii, int64_t *out) {
    const int r = k / 2, W1 = w + 1;
    memset(ii, 0, (size_t)W1 * sizeof(int64_t));
    for (int y = 0; y < h; y++) {
        int64_t run = 0;
        int64_t *row = ii + (size_t)(y + 1) * W1, *up = row - W1;
        row[0] = 0;
        for (int x = 0; x < w; x++) {
            run += a[(size_t)y * w + x];
            row[x + 1] = up[x + 1] + run;
        }
    }
    for (int i = 0; i < h; i++) {
        const int y0 = mo_imax(0, i - r), y1 = mo_imin(h, i + r + 1);
        for (int j = 0; j < w; j++) {
            const int x0 = mo_imax(0, j - r), x1 = mo_imin(w, j + r + 1);
            out[(size_t)i * w + j] = ii[(size_t)y1 * W1 + x1] - ii[(size_t)y0 * W1 + x1] - ii[(size_t)y1 * W1 + x0]
                                     + ii[(size_t)y0 * W1 + x0];
        }
    }
}

/* _ring_clipped: the box sums less the guard x guard sums at the centre (guard 0 = the whole box) */
static void mo_ring_clipped(const int64_t *a, int h, int w, int box, int guard, int64_t *ii, int64_t *g,
                            int64_t *out) {
    mo_box_clipped(a, h, w, box, ii, out);
    if (guard > 0) {                     /* only where the whole box lies inside the grid (see _ring_clipped) */
        const int r = box / 2;
        mo_box_clipped(a, h, w, guard, ii, g);
        for (int y = r; y < h - r; y++)
            for (int x = r; x < w - r; x++) out[(size_t)y * w + x] -= g[(size_t)y * w + x];
    }
}

/* threshold_cells (ND as given: the caller's np.maximum(ND, 1) included); thr uint8, mD float64, ch x cw.
   The neighbourhood is the box less its guard x guard centre: a target never raises its own bar. */
TBOT_API int tbot_motion_threshold(const int64_t *SD, const int64_t *ND, const int64_t *SA, const int64_t *NA,
                                   int32_t ch, int32_t cw, double sigma_n, double t_min_dn, double noise_k,
                                   double cfar_gain, double texture_gain, double abs_to_sigma, int32_t box,
                                   int32_t guard, uint8_t *thr, double *mD) {
    const size_t n = (size_t)ch * cw;
    int64_t *b = (int64_t *)malloc((5 * n + (size_t)(ch + 1) * (cw + 1)) * sizeof(int64_t));
    if (!b) return -1;
    int64_t *S = b, *N = b + n, *sa = b + 2 * n, *na = b + 3 * n, *g = b + 4 * n, *ii = b + 5 * n;
    mo_ring_clipped(SD, ch, cw, box, guard, ii, g, S);
    mo_ring_clipped(ND, ch, cw, box, guard, ii, g, N);
    mo_ring_clipped(SA, ch, cw, box, guard, ii, g, sa);
    mo_ring_clipped(NA, ch, cw, box, guard, ii, g, na);
    const double sig = isfinite(sigma_n) ? sigma_n : 0.0;
    const double fl = mo_pymax(t_min_dn, noise_k * sig);
    const double base = sig / abs_to_sigma;
    for (size_t i = 0; i < n; i++) {
        const double md = (double)S[i] / (double)(N[i] > 1 ? N[i] : 1);
        const double ma = na[i] > 0 ? (double)sa[i] / (double)(na[i] > 1 ? na[i] : 1) : 0.0;
        const double e = ma - base;
        const double exc = 0.0 >= e ? 0.0 : e;                    /* np.maximum(0.0, e) */
        double t = floor(fl + cfar_gain * md + texture_gain * exc);
        t = t < 0.0 ? 0.0 : t > 255.0 ? 255.0 : t;
        thr[i] = (uint8_t)t;
        mD[i] = md;
    }
    free(b);
    return 0;
}

/* compare_cells: binary = 255 where D > thr[y / cell, x / cell] */
TBOT_API void tbot_motion_compare(const uint8_t *D, int32_t h, int32_t w, const uint8_t *thr, int32_t cw,
                                  int32_t cell, uint8_t *binary) {
    uint8_t *big = (uint8_t *)malloc((size_t)(w > 0 ? w : 1));
    if (!big) return;
    for (int y = 0; y < h; y++) {
        if (y % cell == 0) {                     /* the thresholds of this row of cells, one per column */
            const uint8_t *t = thr + (size_t)(y / cell) * cw;
            for (int x = 0; x < w; x++) big[x] = t[x / cell];
        }
        const uint8_t *restrict d = D + (size_t)y * w;
        const uint8_t *restrict t = big;
        uint8_t *restrict o = binary + (size_t)y * w;
        for (int x = 0; x < w; x++) o[x] = d[x] > t[x] ? 255 : 0;
    }
    free(big);
}

/* sum and count of |a - b| over x < n where (va & vb) != 0 */
static inline void mo_sad_run(const uint8_t *restrict a, const uint8_t *restrict b, const uint8_t *restrict va,
                              const uint8_t *restrict vb, int n, int64_t *s, int64_t *c) {
    int x = 0;
    uint64_t ss = 0, cc = 0;
#if defined(__aarch64__)
    while (x + 16 <= n) {
        uint16x8_t as = vdupq_n_u16(0), ac = vdupq_n_u16(0);
        const int lim = n - x > 16 * 128 ? x + 16 * 128 : n;   /* <= 128 steps: a u16 lane holds 65280 */
        for (; x + 16 <= lim; x += 16) {
            const uint8x16_t m = vtstq_u8(vld1q_u8(va + x), vld1q_u8(vb + x));   /* (va & vb) != 0 */
            const uint8x16_t d = vandq_u8(vabdq_u8(vld1q_u8(a + x), vld1q_u8(b + x)), m);
            as = vpadalq_u8(as, d);
            ac = vpadalq_u8(ac, vshrq_n_u8(m, 7));
        }
        ss += vaddlvq_u16(as);
        cc += vaddlvq_u16(ac);
    }
#endif
    for (; x < n; x++) {
        const unsigned m = (va[x] & vb[x]) ? 1u : 0u;
        const unsigned d = a[x] > b[x] ? (unsigned)(a[x] - b[x]) : (unsigned)(b[x] - a[x]);
        ss += m ? d : 0u;
        cc += m;
    }
    *s += (int64_t)ss;
    *c += (int64_t)cc;
}

/* one shift (sx, sy) of sad_surface over the region [x0, x1) x [y0, y1) */
static void mo_sad_one(const uint8_t *cur, const uint8_t *ref, const uint8_t *vcur, const uint8_t *vref, int h, int w,
                       int sx, int sy, int x0, int y0, int x1, int y1, int64_t *s, int64_t *c) {
    *s = 0; *c = 0;
    const int ya = mo_imax(y0, -sy), yb = mo_imin(y1, h - sy);
    const int xa = mo_imax(x0, -sx), xb = mo_imin(x1, w - sx);
    if (yb <= ya || xb <= xa) return;
    const int n = xb - xa;
#if defined(__aarch64__)
    /* u16 lanes: each step adds at most 2 x 255 (16-lane) or 255 (8-lane tail) to a lane: flush before 120 steps */
    uint16x8_t as = vdupq_n_u16(0), ac = vdupq_n_u16(0);
    int steps = 0;
    uint64_t ss = 0, cc = 0;
    const int n16 = n & ~15, n8 = (n - n16) >= 8 ? 8 : 0;
    for (int y = ya; y < yb; y++) {
        const size_t p = (size_t)y * w + xa, q = (size_t)(y + sy) * w + xa + sx;
        const uint8_t *a = cur + p, *b = ref + q, *va = vcur + p, *vb = vref + q;
        int x = 0;
        for (; x < n16; x += 16) {
            const uint8x16_t m = vtstq_u8(vld1q_u8(va + x), vld1q_u8(vb + x));
            const uint8x16_t d = vandq_u8(vabdq_u8(vld1q_u8(a + x), vld1q_u8(b + x)), m);
            as = vpadalq_u8(as, d);
            ac = vpadalq_u8(ac, vshrq_n_u8(m, 7));
            if (++steps == 120) {
                ss += vaddlvq_u16(as); cc += vaddlvq_u16(ac);
                as = vdupq_n_u16(0); ac = vdupq_n_u16(0); steps = 0;
            }
        }
        if (n8) {
            const uint8x8_t m = vtst_u8(vld1_u8(va + x), vld1_u8(vb + x));
            const uint8x8_t d = vand_u8(vabd_u8(vld1_u8(a + x), vld1_u8(b + x)), m);
            as = vaddw_u8(as, d);
            ac = vaddw_u8(ac, vshr_n_u8(m, 7));
            x += 8;
            if (++steps >= 120) {
                ss += vaddlvq_u16(as); cc += vaddlvq_u16(ac);
                as = vdupq_n_u16(0); ac = vdupq_n_u16(0); steps = 0;
            }
        }
        for (; x < n; x++) {
            const unsigned m = (va[x] & vb[x]) ? 1u : 0u;
            const unsigned d = a[x] > b[x] ? (unsigned)(a[x] - b[x]) : (unsigned)(b[x] - a[x]);
            ss += m ? d : 0u;
            cc += m;
        }
    }
    ss += vaddlvq_u16(as);
    cc += vaddlvq_u16(ac);
    *s = (int64_t)ss;
    *c = (int64_t)cc;
#else
    for (int y = ya; y < yb; y++) {
        const size_t p = (size_t)y * w + xa, q = (size_t)(y + sy) * w + xa + sx;
        mo_sad_run(cur + p, ref + q, vcur + p, vref + q, n, s, c);
    }
#endif
}

/* sad_surface: sums, cnts (2r + 1) x (2r + 1), shift (cx + i - r, cy + j - r) at [j][i] */
TBOT_API void tbot_motion_sad(const uint8_t *cur, const uint8_t *ref, const uint8_t *vcur, const uint8_t *vref,
                              int32_t h, int32_t w, int32_t cx, int32_t cy, int32_t r, int32_t x0, int32_t y0,
                              int32_t x1, int32_t y1, int64_t *sums, int64_t *cnts) {
    const int n = 2 * r + 1;
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
            mo_sad_one(cur, ref, vcur, vref, h, w, cx + i - r, cy + j - r, x0, y0, x1, y1,
                       sums + (size_t)j * n + i, cnts + (size_t)j * n + i);
}

/* sad_cells: sums, cnts (rows, cols, 2r + 1, 2r + 1) for the grid with row edges ys (rows + 1), column edges xs */
TBOT_API void tbot_motion_sad_cells(const uint8_t *cur, const uint8_t *ref, const uint8_t *vcur, const uint8_t *vref,
                                    int32_t h, int32_t w, int32_t r, const int32_t *ys, int32_t rows, const int32_t *xs,
                                    int32_t cols, int64_t *sums, int64_t *cnts) {
    const int n = 2 * r + 1, nn = n * n;
    for (int a = 0; a < rows; a++)
        for (int b = 0; b < cols; b++) {
            const size_t o = ((size_t)a * cols + b) * nn;
            tbot_motion_sad(cur, ref, vcur, vref, h, w, 0, 0, r, xs[b], ys[a], xs[b + 1], ys[a + 1], sums + o,
                            cnts + o);
        }
}

/* ------------------------------------------------------------------------------------- registration */
typedef struct {
    int32_t hc, wc, hf, wf, block, r_c;
    int32_t gr, gc, field_r, use_field;          /* local field: grid, SAD radius; use_field: ok and use_field */
    int32_t np_chunk;                             /* NumPy's reduction run length (mo_sum) */
    int32_t need_x, need_y, pad_;                 /* axes the fine search may re-centre along */
    const uint8_t *cur_c, *cur_f, *tex_f, *ref_c, *ref_f;
    const uint8_t *srcv_c, *srcv_f, *curv_c, *curv_f;
    uint8_t *wrc, *vrc, *wrf, *vrf;                   /* scratch at the coarse / fine size */
    int32_t *dx, *dy;                             /* hf x nbf */
    double *rx, *ry;                              /* hf x nbf (phase 2 leaves the final residuals here) */
    const int32_t *ys, *xs;                       /* field grid edges, gr + 1 / gc + 1 */
    double *field;                                /* out: gr x gc x 2 */
    uint8_t *field_ok;                            /* out: gr x gc */
    double lc[9], lf[9], lm[9];                   /* level matrices (phase 1: coarse, fine; phase 2: final) */
    double min_overlap_frac, black_dn, gain_min, gain_max, reg_min_response, fit_std_min, cell_texture_frac;
    /* phase 1 out */
    int32_t early, edge, curv, n_ovl, n_tex;
    int32_t g0[2];
    double depth, delta[2];
    /* phase 2 out */
    int32_t gq[2];
    double a, b;
} tbot_motion_reg;

static double mo_parabola(double sm, double s0, double sp, int *ok) {   /* motion._parabola */
    const double den = sm - 2.0 * s0 + sp;
    if (den <= 0.0) { *ok = 0; return 0.0; }
    *ok = 1;
    return mo_pymin(0.5, mo_pymax(-0.5, 0.5 * (sm - sp) / den));
}

static int mo_argmin(const double *a, int n) {   /* np.argmin: first minimum */
    int k = 0;
    for (int i = 1; i < n; i++)
        if (a[i] < a[k]) k = i;
    return k;
}

TBOT_API int tbot_motion_reg_size(void) { return (int)sizeof(tbot_motion_reg); }

TBOT_API int tbot_motion_reg1(tbot_motion_reg *p) {
    const int hc = p->hc, wc = p->wc, hf = p->hf, wf = p->wf, blk = p->block;
    double res_c[2], res_f[2];
    p->early = 0;
    /* coarse: block warp by the prior, ratio-of-means gain, SAD search */
    mo_offsets(p->lc, 3, hc, wc, blk, NULL, NULL, p->dx, p->dy, p->rx, p->ry, res_c, p->np_chunk);
    tbot_motion_warp(p->ref_c, p->srcv_c, hc, wc, p->dx, p->dy, blk, p->wrc, p->vrc);
    const size_t nc = (size_t)hc * wc;
    int64_t nboth = 0, sc = 0, sr = 0;
    for (size_t i = 0; i < nc; i++)
        if (p->vrc[i] && p->curv_c[i]) { nboth++; sc += p->cur_c[i]; sr += p->wrc[i]; }
    if ((double)nboth < p->min_overlap_frac * (double)nc) { p->early = 1; return 0; }
    const double mc = (double)sc / (double)nboth, mr = (double)sr / (double)nboth;
    const double a0 = mo_pymin(mo_pymax((mc - p->black_dn) / mo_pymax(mr - p->black_dn, 1.0), p->gain_min),
                               p->gain_max);
    p->g0[0] = (int32_t)nearbyint(a0 * 256.0);
    p->g0[1] = (int32_t)nearbyint(p->black_dn * (1.0 - a0) * 256.0) + 128;
    tbot_motion_gain(p->wrc, (int64_t)nc, p->g0[0], p->g0[1], p->wrc);
    if (p->r_c < 0) return -1;
    const int rc = p->r_c, n = 2 * rc + 1, nn = n * n;
    int64_t *ss = (int64_t *)malloc((size_t)nn * 2 * sizeof(int64_t));
    double *sad = (double *)malloc((size_t)nn * 2 * sizeof(double));
    if (!ss || !sad) { free(ss); free(sad); return -1; }
    int64_t *cn = ss + nn;
    double *fin = sad + nn;
    tbot_motion_sad(p->cur_c, p->wrc, p->curv_c, p->vrc, hc, wc, 0, 0, rc, 0, 0, wc, hc, ss, cn);
    const double need = p->min_overlap_frac * (double)nc;
    int nfin = 0;
    for (int k = 0; k < nn; k++) {
        sad[k] = (double)cn[k] >= need ? (double)ss[k] / (double)(cn[k] > 1 ? cn[k] : 1) : INFINITY;
        if (isfinite(sad[k])) fin[nfin++] = sad[k];
    }
    const int kmin = mo_argmin(sad, nn), j = kmin / n, i = kmin % n;
    const double smin = sad[kmin];
    const double med = nfin ? mo_median(fin, nfin) : 0.0;
    p->depth = isfinite(smin) ? (med - smin) / mo_pymax(med, 1e-6) : 0.0;
    p->edge = rc > 0 && (abs(i - rc) == rc || abs(j - rc) == rc);
    free(ss); free(sad);
    const double dcx = (double)(i - rc) - res_c[0], dcy = (double)(j - rc) - res_c[1];
    /* fine: +-1 around the coarse result, recentred up to twice, parabolic sub-pixel */
    mo_offsets(p->lf, 3, hf, wf, blk, NULL, NULL, p->dx, p->dy, p->rx, p->ry, res_f, p->np_chunk);
    tbot_motion_warp(p->ref_f, p->srcv_f, hf, wf, p->dx, p->dy, blk, p->wrf, p->vrf);
    const size_t nf = (size_t)hf * wf;
    tbot_motion_gain(p->wrf, (int64_t)nf, p->g0[0], p->g0[1], p->wrf);
    int64_t cx = (int64_t)floor(2.0 * dcx + res_f[0] + 0.5), cy = (int64_t)floor(2.0 * dcy + res_f[1] + 0.5);
    int okx = 0, oky = 0, allfin = 0;
    double subx = 0.0, suby = 0.0;
    int64_t memo[7 * 7 * 2];                              /* shifts within +-3 of the first centre */
    uint8_t have[49];
    memset(have, 0, sizeof have);
    const int64_t cx0 = cx, cy0 = cy;
    for (int it = 0; it < 3; it++) {                     /* FINE_RECENTRE + 1 */
        int64_t fs[9], fn[9];
        double fsad[9];
        for (int k = 0; k < 9; k++) {
            const int64_t sx = cx + k % 3 - 1, sy = cy + k / 3 - 1;
            const int64_t ix = sx - cx0 + 3, iy = sy - cy0 + 3;
            const int cache = ix >= 0 && ix < 7 && iy >= 0 && iy < 7;
            const int slot = cache ? (int)(iy * 7 + ix) : 0;
            if (cache && have[slot]) { fs[k] = memo[2 * slot]; fn[k] = memo[2 * slot + 1]; continue; }
            mo_sad_one(p->cur_f, p->wrf, p->curv_f, p->vrf, hf, wf, (int)sx, (int)sy, 0, 0, wf, hf, fs + k, fn + k);
            if (cache) { have[slot] = 1; memo[2 * slot] = fs[k]; memo[2 * slot + 1] = fn[k]; }
        }
        for (int k = 0; k < 9; k++) fsad[k] = fn[k] > 0 ? (double)fs[k] / (double)(fn[k] > 1 ? fn[k] : 1) : INFINITY;
        const int km = mo_argmin(fsad, 9), jj = km / 3, ii = km % 3;
        if (isfinite(fsad[km]) && ((p->need_x && (ii == 0 || ii == 2)) || (p->need_y && (jj == 0 || jj == 2)))) {
            if (p->need_x) cx += ii - 1;
            if (p->need_y) cy += jj - 1;
            continue;
        }
        subx = mo_parabola(fsad[3], fsad[4], fsad[5], &okx);
        suby = mo_parabola(fsad[1], fsad[4], fsad[7], &oky);
        allfin = 1;
        for (int k = 0; k < 9; k++) allfin &= isfinite(fsad[k]) != 0;
        break;
    }
    p->curv = okx | (oky << 1) | (allfin << 2);   /* the parabola in x, in y, the last surface all finite */
    /* delta_f (fine px, ref space): the caller multiplies by rd */
    p->delta[0] = ((double)cx + subx) - res_f[0];
    p->delta[1] = ((double)cy + suby) - res_f[1];
    /* texture guard counts */
    int32_t novl = 0, ntex = 0;
    for (size_t k = 0; k < nf; k++) {
        const int o = p->vrf[k] != 0 && p->curv_f[k] != 0;
        novl += o;
        ntex += o && p->tex_f[k] != 0;
    }
    p->n_ovl = novl;
    p->n_tex = ntex;
    return 0;
}

/* photometric_fit's percentile ("linear") of a 256-bin histogram with n values */
static double mo_hist_kth(const int64_t *hist, int64_t k) {
    int64_t c = 0;
    for (int v = 0; v < 256; v++) {
        c += hist[v];
        if (c > k) return (double)v;
    }
    return 255.0;
}

static double mo_percentile(const int64_t *hist, int64_t n, double q) {
    const double vi = (double)(n - 1) * q;
    double prev = floor(vi);
    int64_t ip = (int64_t)prev, in = ip + 1;
    if (vi >= (double)(n - 1)) { ip = n - 1; in = n - 1; prev = -1.0; }
    if (vi < 0.0) { ip = 0; in = 0; prev = 0.0; }
    const double gamma = vi - prev;
    const double a = mo_hist_kth(hist, ip), b = mo_hist_kth(hist, in);
    const double diff = (double)(uint8_t)((int)b - (int)a);
    if (gamma >= 0.5) return b - diff * (1.0 - gamma);
    return a + diff * gamma;
}

TBOT_API void tbot_motion_photometric(const uint8_t *cur, const uint8_t *ref, const uint8_t *valid, int32_t h,
                                      int32_t w, double black_dn, double gain_min, double gain_max, double fit_std_min,
                                      double *ab, int32_t *gq) {
    int64_t hx[256], hy[256], n = 0, sx = 0, sy = 0;
    memset(hx, 0, sizeof hx);
    memset(hy, 0, sizeof hy);
    for (int y = 0; y < h; y += 2) {
        const size_t o = (size_t)y * w;
        for (int x = 0; x < w; x += 2)
            if (valid[o + x]) { hx[ref[o + x]]++; hy[cur[o + x]]++; n++; sx += ref[o + x]; sy += cur[o + x]; }
    }
    double a = 1.0, b = 0.0;
    if (n >= 16) {
        const double q10 = 10.0 / 100.0, q50 = 50.0 / 100.0, q90 = 90.0 / 100.0;
        const double x10 = mo_percentile(hx, n, q10), x50 = mo_percentile(hx, n, q50), x90 = mo_percentile(hx, n, q90);
        const double y10 = mo_percentile(hy, n, q10), y50 = mo_percentile(hy, n, q50), y90 = mo_percentile(hy, n, q90);
        if (x90 - x10 < 2.56 * fit_std_min) {
            const double my = (double)sy / (double)n, mx = (double)sx / (double)n;
            a = (my - black_dn) / mo_pymax(mx - black_dn, 1.0);
            a = mo_pymin(mo_pymax(a, gain_min), gain_max);
            b = black_dn * (1.0 - a);
        } else {
            a = mo_pymin(mo_pymax((y90 - y10) / (x90 - x10), gain_min), gain_max);
            b = y50 - a * x50;
        }
    }
    ab[0] = a;
    ab[1] = b;
    gq[0] = (int32_t)nearbyint(a * 256.0);
    gq[1] = (int32_t)nearbyint(b * 256.0) + 128;
}

/* MotionDetector._local_field on the gained fine warp (wf, vf) with the final residuals (rx, ry) */
static int mo_local_field(tbot_motion_reg *p, double rd) {
    const int h = p->hf, w = p->wf, gr = p->gr, gc = p->gc, R = p->field_r, n = 2 * R + 1, nn = n * n;
    const int nb = (w + p->block - 1) / p->block;
    const int ncell = gr * gc;
    double *f = (double *)calloc((size_t)ncell * 2, sizeof(double));
    int64_t *nvs = (int64_t *)calloc((size_t)ncell * 2, sizeof(int64_t));
    double *tmp = (double *)malloc(((size_t)h * nb + (size_t)nn) * sizeof(double));
    if (!f || !nvs || !tmp) { free(f); free(nvs); free(tmp); return -1; }
    int64_t *nts = nvs + ncell;
    memset(p->field_ok, 0, (size_t)ncell);
    for (int a = 0; a < gr; a++)
        for (int y = p->ys[a]; y < p->ys[a + 1]; y++) {
            const uint8_t *v = p->vrf + (size_t)y * w, *c = p->curv_f + (size_t)y * w, *t = p->tex_f + (size_t)y * w;
            for (int b = 0; b < gc; b++) {
                int64_t s = 0, st = 0;
                for (int x = p->xs[b]; x < p->xs[b + 1]; x++) {
                    const int o = (v[x] & c[x]) != 0;
                    s += o;
                    st += o && t[x] != 0;
                }
                nvs[a * gc + b] += s;
                nts[a * gc + b] += st;
            }
        }
    int any = 0;
    uint8_t cand[64];
    for (int a = 0; a < gr; a++)
        for (int b = 0; b < gc; b++) {
            const int k = a * gc + b;
            const double area = 0.5 * (double)(p->ys[a + 1] - p->ys[a]) * (double)(p->xs[b + 1] - p->xs[b]);
            cand[k] = (double)nvs[k] >= area && (double)nts[k] >= p->cell_texture_frac * (double)nvs[k];
            any |= cand[k];
        }
    if (!any) {
        memset(p->field, 0, (size_t)ncell * 2 * sizeof(double));
        free(f); free(nvs); free(tmp);
        return 0;
    }
    for (int a = 0; a < gr; a++)
        for (int b = 0; b < gc; b++) {
            const int k = a * gc + b;
            if (!cand[k]) continue;
            const int x0 = p->xs[b], x1 = p->xs[b + 1], y0 = p->ys[a], y1 = p->ys[a + 1];
            int64_t ss[81], cc[81];
            double sad[81];
            tbot_motion_sad(p->cur_f, p->wrf, p->curv_f, p->vrf, h, w, 0, 0, R, x0, y0, x1, y1, ss, cc);
            const double half = 0.5 * (double)nvs[k];
            int allfin = 1;
            for (int q = 0; q < nn; q++) {
                sad[q] = (double)cc[q] >= half ? (double)ss[q] / (double)(cc[q] > 1 ? cc[q] : 1) : INFINITY;
                allfin &= isfinite(sad[q]) != 0;
            }
            const int km = mo_argmin(sad, nn), j = km / n, i = km % n;
            if (!allfin || i == 0 || i == 2 * R || j == 0 || j == 2 * R) continue;
            double srt[81];
            memcpy(srt, sad, (size_t)nn * sizeof(double));
            const double med = mo_median(srt, nn);
            if ((med - sad[km]) / mo_pymax(med, 1e-6) < p->reg_min_response) continue;
            int okx, oky;
            const double sx = mo_parabola(sad[j * n + i - 1], sad[km], sad[j * n + i + 1], &okx);
            const double sy = mo_parabola(sad[(j - 1) * n + i], sad[km], sad[(j + 1) * n + i], &oky);
            if (!(okx && oky)) continue;
            /* mean residual of the blocks whose centre lies in the cell: rows y0..y1, those columns, F order */
            double mrx = 0.0, mry = 0.0;
            int m = 0;
            for (int bb = 0; bb < nb; bb++) {
                const int bx0 = bb * p->block, bx1 = mo_imin(bx0 + p->block, w);
                const double xc = (double)(bx0 + bx1 - 1) * 0.5;
                if (!(xc >= (double)x0 && xc < (double)x1)) continue;
                for (int y = y0; y < y1; y++) tmp[m++] = p->rx[(size_t)y * nb + bb];
            }
            if (m > 0) {
                mrx = mo_mean(tmp, m, p->np_chunk);
                m = 0;
                for (int bb = 0; bb < nb; bb++) {
                    const int bx0 = bb * p->block, bx1 = mo_imin(bx0 + p->block, w);
                    const double xc = (double)(bx0 + bx1 - 1) * 0.5;
                    if (!(xc >= (double)x0 && xc < (double)x1)) continue;
                    for (int y = y0; y < y1; y++) tmp[m++] = p->ry[(size_t)y * nb + bb];
                }
                mry = mo_mean(tmp, m, p->np_chunk);
            }
            f[2 * k] = (((double)(i - R) + sx) - mrx) * rd;
            f[2 * k + 1] = (((double)(j - R) + sy) - mry) * rd;
            p->field_ok[k] = 1;
        }
    /* 3 x 3 median over the grid, clipped at its edges */
    for (int a = 0; a < gr; a++)
        for (int b = 0; b < gc; b++)
            for (int c = 0; c < 2; c++) {
                double v[9];
                int m = 0;
                for (int aa = mo_imax(0, a - 1); aa < mo_imin(gr, a + 2); aa++)
                    for (int bb = mo_imax(0, b - 1); bb < mo_imin(gc, b + 2); bb++) v[m++] = f[2 * (aa * gc + bb) + c];
                mo_sort_doubles(v, m);
                p->field[2 * (a * gc + b) + c] = (m & 1) ? (0.0 + v[m / 2]) / 1.0 : ((0.0 + v[m / 2 - 1]) + v[m / 2]) / 2.0;
            }
    free(f); free(nvs); free(tmp);
    return 0;
}

/* phase 2: warp by lm (final transform at the fine level), photometric fit, and with use_field the local field.
 * rd = reg_downscale (fine px -> full-res px). Returns 0, -1 (memory) */
TBOT_API int tbot_motion_reg2(tbot_motion_reg *p, double rd) {
    const int hf = p->hf, wf = p->wf;
    const size_t nf = (size_t)hf * wf;
    mo_offsets(p->lm, 3, hf, wf, p->block, NULL, NULL, p->dx, p->dy, p->rx, p->ry, NULL, 0);
    tbot_motion_warp(p->ref_f, p->srcv_f, hf, wf, p->dx, p->dy, p->block, p->wrf, p->vrf);
    uint8_t *fv = (uint8_t *)malloc(nf);
    if (!fv) return -1;
    for (size_t k = 0; k < nf; k++) fv[k] = p->vrf[k] != 0 && p->curv_f[k] != 0;
    double ab[2];
    tbot_motion_photometric(p->cur_f, p->wrf, fv, hf, wf, p->black_dn, p->gain_min, p->gain_max, p->fit_std_min, ab,
                            p->gq);
    free(fv);
    p->a = ab[0];
    p->b = ab[1];
    if (!p->use_field) return 0;
    if (p->gr * p->gc > 64 || p->field_r > 4) return -1;
    tbot_motion_gain(p->wrf, (int64_t)nf, p->gq[0], p->gq[1], p->wrf);
    return mo_local_field(p, rd);
}

/* ---------------------------------------------------------------------------------------- the kernel */
/* _field_extra (bilinear field at the working warp blocks, / s) into ex, ey (h x nb): tables from Python */
TBOT_API void tbot_motion_field_extra(const double *field, int32_t gr, int32_t gc, int32_t h, int32_t nb,
                                      const int32_t *x0, const int32_t *x1, const double *tx, const int32_t *y0,
                                      const int32_t *y1, const double *ty, double s, double *ex, double *ey) {
    (void)gr;
    for (int c = 0; c < 2; c++) {
        double *out = c ? ey : ex;
        for (int r = 0; r < h; r++) {
            const double *F0 = field + (size_t)y0[r] * gc * 2, *F1 = field + (size_t)y1[r] * gc * 2;
            const double wy0 = 1.0 - ty[r], wy1 = ty[r];
            for (int b = 0; b < nb; b++) {
                const double wx0 = 1.0 - tx[b], wx1 = tx[b];
                const double top = F0[2 * x0[b] + c] * wx0 + F0[2 * x1[b] + c] * wx1;
                const double bot = F1[2 * x0[b] + c] * wx0 + F1[2 * x1[b] + c] * wx1;
                out[(size_t)r * nb + b] = (top * wy0 + bot * wy1) / s;
            }
        }
    }
}

/* block_offsets(m, h, w, extra=_field_extra(field)) with field NULL or all zero meaning no extra */
TBOT_API void tbot_motion_offsets(const double *m, int32_t h, int32_t w, int32_t block, const double *field,
                                  int32_t gr, int32_t gc, const int32_t *x0, const int32_t *x1, const double *tx,
                                  const int32_t *y0, const int32_t *y1, const double *ty, double s, double *ex,
                                  double *ey, int32_t *dx, int32_t *dy, double *rx, double *ry) {
    const int nb = (w + block - 1) / block;
    int any = 0;
    if (field)
        for (int k = 0; k < gr * gc * 2; k++) any |= field[k] != 0.0;
    if (any) tbot_motion_field_extra(field, gr, gc, h, nb, x0, x1, tx, y0, y1, ty, s, ex, ey);
    mo_offsets(m, 3, h, w, block, any ? ex : NULL, any ? ey : NULL, dx, dy, rx, ry, NULL, 0);
}

typedef struct {
    int32_t h, w, block, env, cell, row_step, cfar_box, cfar_guard;
    int64_t ga, oa, gb, ob;
    const uint8_t *cur, *ref_a, *ref_b, *src_valid, *mask;    /* mask may be NULL */
    const int32_t *dxa, *dya, *dxb, *dyb;
    uint8_t *wa, *wb, *va, *vb, *D, *binary, *thr;
    double *mD;
    int64_t *SD, *ND, *SA, *NA;                               /* ch x cw each */
    uint8_t *scratch;                                         /* 6 x h x w */
    double t_min_dn, noise_k, cfar_gain, texture_gain, abs_to_sigma, noise_quantile, noise_min_fill;
    double sigma_n;                                           /* out */
} tbot_motion_kparams;

TBOT_API int tbot_motion_kparams_size(void) { return (int)sizeof(tbot_motion_kparams); }

/* rows y - a .. y + bb of src (clamped: the same max / min as the window clipped to the frame) -> vmax, vmin */
static void mo_vert(const uint8_t *src, int h, int w, int y, int a, int bb, uint8_t *restrict vmax,
                    uint8_t *restrict vmin) {
    const uint8_t *r0 = src + (size_t)y * w;
    memcpy(vmax, r0, (size_t)w);
    memcpy(vmin, r0, (size_t)w);
    for (int o = -a; o <= bb; o++) {
        if (o == 0) continue;
        const int yy = y + o < 0 ? 0 : y + o >= h ? h - 1 : y + o;
        const uint8_t *restrict r = src + (size_t)yy * w;
        for (int x = 0; x < w; x++) {
            vmax[x] = r[x] > vmax[x] ? r[x] : vmax[x];
            vmin[x] = r[x] < vmin[x] ? r[x] : vmin[x];
        }
    }
}

/* columns x - a .. x + bb (clamped) */
static void mo_horiz(const uint8_t *restrict v, int w, int a, int bb, int is_max, uint8_t *restrict out) {
    memcpy(out, v, (size_t)w);
    for (int o = -a; o <= bb; o++) {
        if (o == 0) continue;
        const int x0 = mo_imax(0, -o), x1 = mo_imin(w, w - o);   /* interior: x + o inside */
        if (is_max) {
            for (int x = x0; x < x1; x++) out[x] = v[x + o] > out[x] ? v[x + o] : out[x];
            for (int x = 0; x < x0 && x < w; x++) { const uint8_t q = v[0]; out[x] = q > out[x] ? q : out[x]; }
            for (int x = x1 > 0 ? x1 : 0; x < w; x++) { const uint8_t q = v[w - 1]; out[x] = q > out[x] ? q : out[x]; }
        } else {
            for (int x = x0; x < x1; x++) out[x] = v[x + o] < out[x] ? v[x + o] : out[x];
            for (int x = 0; x < x0 && x < w; x++) { const uint8_t q = v[0]; out[x] = q < out[x] ? q : out[x]; }
            for (int x = x1 > 0 ? x1 : 0; x < w; x++) { const uint8_t q = v[w - 1]; out[x] = q < out[x] ? q : out[x]; }
        }
    }
}

/* envelope residual, validity and the four cell statistics in one pass over the rows (row buffers in L1).
 * cell <= 256 (uint16 column sums). scratch: 9 rows of w bytes + 4 rows of w uint16. */
static void mo_kernel_rows(const tbot_motion_kparams *p, uint8_t *scratch) {
    const int h = p->h, w = p->w, cell = p->cell, rs = p->row_step, k = p->env, a = k / 2, bb = k - 1 - a;
    const int ch = (h + cell - 1) / cell, cw = (w + cell - 1) / cell;
    uint8_t *vmx = scratch, *vmn = vmx + w, *hiA = vmn + w, *loA = hiA + w, *hiB = loA + w, *loB = hiB + w;
    uint16_t *cD = (uint16_t *)(((uintptr_t)(loB + w) + 15) & ~(uintptr_t)15), *cN = cD + w, *cS = cN + w, *cA = cS + w;
    memset(p->SD, 0, (size_t)ch * cw * sizeof(int64_t));
    memset(p->ND, 0, (size_t)ch * cw * sizeof(int64_t));
    memset(p->SA, 0, (size_t)ch * cw * sizeof(int64_t));
    memset(p->NA, 0, (size_t)ch * cw * sizeof(int64_t));
    memset(cD, 0, 4 * (size_t)w * sizeof(uint16_t));
#if defined(__aarch64__)
    const int fast3 = k == 3 && w >= 18;
    /* padded vertical rows (index x + 1 = pixel x; [0] and [w + 1] repeat the border): 4 x (w + 2) bytes */
    uint8_t *pv = (uint8_t *)(cA + w);
    uint8_t *pA = pv, *qA = pA + w + 2, *pB = qA + w + 2, *qB = pB + w + 2;
#endif
    for (int y = 0; y < h; y++) {
#if defined(__aarch64__)
        if (fast3) {
            const uint8_t *a0 = p->wa + (size_t)(y > 0 ? y - 1 : 0) * w, *a1 = p->wa + (size_t)y * w,
                          *a2 = p->wa + (size_t)(y + 1 < h ? y + 1 : h - 1) * w;
            const uint8_t *b0 = p->wb + (size_t)(y > 0 ? y - 1 : 0) * w, *b1 = p->wb + (size_t)y * w,
                          *b2 = p->wb + (size_t)(y + 1 < h ? y + 1 : h - 1) * w;
            int x = 0;
            for (; x + 16 <= w; x += 16) {
                const uint8x16_t u0 = vld1q_u8(a0 + x), u1 = vld1q_u8(a1 + x), u2 = vld1q_u8(a2 + x);
                const uint8x16_t t0 = vld1q_u8(b0 + x), t1 = vld1q_u8(b1 + x), t2 = vld1q_u8(b2 + x);
                vst1q_u8(pA + 1 + x, vmaxq_u8(vmaxq_u8(u0, u1), u2));
                vst1q_u8(qA + 1 + x, vminq_u8(vminq_u8(u0, u1), u2));
                vst1q_u8(pB + 1 + x, vmaxq_u8(vmaxq_u8(t0, t1), t2));
                vst1q_u8(qB + 1 + x, vminq_u8(vminq_u8(t0, t1), t2));
            }
            for (; x < w; x++) {
                uint8_t m0 = a0[x], m1 = a1[x], m2 = a2[x];
                pA[1 + x] = m0 > m1 ? (m0 > m2 ? m0 : m2) : (m1 > m2 ? m1 : m2);
                qA[1 + x] = m0 < m1 ? (m0 < m2 ? m0 : m2) : (m1 < m2 ? m1 : m2);
                m0 = b0[x]; m1 = b1[x]; m2 = b2[x];
                pB[1 + x] = m0 > m1 ? (m0 > m2 ? m0 : m2) : (m1 > m2 ? m1 : m2);
                qB[1 + x] = m0 < m1 ? (m0 < m2 ? m0 : m2) : (m1 < m2 ? m1 : m2);
            }
            pA[0] = pA[1]; qA[0] = qA[1]; pB[0] = pB[1]; qB[0] = qB[1];
            pA[w + 1] = pA[w]; qA[w + 1] = qA[w]; pB[w + 1] = pB[w]; qB[w + 1] = qB[w];
            /* the horizontal step for the scalar tail (and nothing else) */
            for (int xx = w & ~15; xx < w; xx++) {
                const uint8_t *P0 = pA + xx, *Q0 = qA + xx, *P1 = pB + xx, *Q1 = qB + xx;
                uint8_t m = P0[0] > P0[1] ? P0[0] : P0[1]; hiA[xx] = m > P0[2] ? m : P0[2];
                m = Q0[0] < Q0[1] ? Q0[0] : Q0[1]; loA[xx] = m < Q0[2] ? m : Q0[2];
                m = P1[0] > P1[1] ? P1[0] : P1[1]; hiB[xx] = m > P1[2] ? m : P1[2];
                m = Q1[0] < Q1[1] ? Q1[0] : Q1[1]; loB[xx] = m < Q1[2] ? m : Q1[2];
            }
        } else
#endif
        {
            mo_vert(p->wa, h, w, y, a, bb, vmx, vmn);
            mo_horiz(vmx, w, a, bb, 1, hiA);
            mo_horiz(vmn, w, a, bb, 0, loA);
            mo_vert(p->wb, h, w, y, a, bb, vmx, vmn);
            mo_horiz(vmx, w, a, bb, 1, hiB);
            mo_horiz(vmn, w, a, bb, 0, loB);
        }
        const size_t o = (size_t)y * w;
        const uint8_t *restrict c = p->cur + o, *restrict va = p->va + o, *restrict vb = p->vb + o;
        const uint8_t *restrict ma = p->mask ? p->mask + o : NULL, *restrict wa = p->wa + o;
        uint8_t *restrict D = p->D + o;
        int x0 = 0;           /* pixels before x0 are done (NEON), including their column sums */
#if defined(__aarch64__)
        {
            const int do_a = y % rs == 0;
            const uint8x16_t one = vdupq_n_u8(1);
            for (; x0 + 16 <= w; x0 += 16) {
                const int x = x0;
                const uint8x16_t q = vld1q_u8(c + x);
                uint8x16_t hA, lA, hB, lB;
                if (fast3) {          /* padded rows: x - 1, x, x + 1 are at x, x + 1, x + 2 */
                    hA = vmaxq_u8(vmaxq_u8(vld1q_u8(pA + x), vld1q_u8(pA + x + 1)), vld1q_u8(pA + x + 2));
                    lA = vminq_u8(vminq_u8(vld1q_u8(qA + x), vld1q_u8(qA + x + 1)), vld1q_u8(qA + x + 2));
                    hB = vmaxq_u8(vmaxq_u8(vld1q_u8(pB + x), vld1q_u8(pB + x + 1)), vld1q_u8(pB + x + 2));
                    lB = vminq_u8(vminq_u8(vld1q_u8(qB + x), vld1q_u8(qB + x + 1)), vld1q_u8(qB + x + 2));
                } else {
                    hA = vld1q_u8(hiA + x); lA = vld1q_u8(loA + x); hB = vld1q_u8(hiB + x); lB = vld1q_u8(loB + x);
                }
                const uint8x16_t b1 = vqsubq_u8(q, hA), b2 = vqsubq_u8(q, hB);
                const uint8x16_t d1 = vqsubq_u8(lA, q), d2 = vqsubq_u8(lB, q);
                const uint8x16_t r = vmaxq_u8(vminq_u8(b1, b2), vminq_u8(d1, d2));
                uint8x16_t v = vandq_u8(vld1q_u8(va + x), vld1q_u8(vb + x));
                if (ma) v = vandq_u8(v, vld1q_u8(ma + x));
                const uint8x16_t dv = vandq_u8(r, v);
                vst1q_u8(D + x, dv);
                const uint8x16_t nz = vandq_u8(vtstq_u8(v, v), one);
                vst1q_u16(cD + x, vaddw_u8(vld1q_u16(cD + x), vget_low_u8(dv)));
                vst1q_u16(cD + x + 8, vaddw_high_u8(vld1q_u16(cD + x + 8), dv));
                vst1q_u16(cN + x, vaddw_u8(vld1q_u16(cN + x), vget_low_u8(nz)));
                vst1q_u16(cN + x + 8, vaddw_high_u8(vld1q_u16(cN + x + 8), nz));
                if (do_a) {
                    const uint8x16_t ad = vandq_u8(vabdq_u8(q, vld1q_u8(wa + x)), v);
                    vst1q_u16(cS + x, vaddw_u8(vld1q_u16(cS + x), vget_low_u8(ad)));
                    vst1q_u16(cS + x + 8, vaddw_high_u8(vld1q_u16(cS + x + 8), ad));
                    vst1q_u16(cA + x, vaddw_u8(vld1q_u16(cA + x), vget_low_u8(nz)));
                    vst1q_u16(cA + x + 8, vaddw_high_u8(vld1q_u16(cA + x + 8), nz));
                }
            }
        }
#endif
        if (ma) {
            for (int x = x0; x < w; x++) {
                const uint8_t q = c[x];
                const uint8_t b1 = q > hiA[x] ? (uint8_t)(q - hiA[x]) : 0, b2 = q > hiB[x] ? (uint8_t)(q - hiB[x]) : 0;
                const uint8_t d1 = loA[x] > q ? (uint8_t)(loA[x] - q) : 0, d2 = loB[x] > q ? (uint8_t)(loB[x] - q) : 0;
                const uint8_t br = b1 < b2 ? b1 : b2, dk = d1 < d2 ? d1 : d2;
                const uint8_t v = (uint8_t)(va[x] & vb[x] & ma[x]);
                D[x] = (uint8_t)((br > dk ? br : dk) & v);
                vmx[x] = v;
            }
        } else {
            for (int x = x0; x < w; x++) {
                const uint8_t q = c[x];
                const uint8_t b1 = q > hiA[x] ? (uint8_t)(q - hiA[x]) : 0, b2 = q > hiB[x] ? (uint8_t)(q - hiB[x]) : 0;
                const uint8_t d1 = loA[x] > q ? (uint8_t)(loA[x] - q) : 0, d2 = loB[x] > q ? (uint8_t)(loB[x] - q) : 0;
                const uint8_t br = b1 < b2 ? b1 : b2, dk = d1 < d2 ? d1 : d2;
                const uint8_t v = (uint8_t)(va[x] & vb[x]);
                D[x] = (uint8_t)((br > dk ? br : dk) & v);
                vmx[x] = v;
            }
        }
        const uint8_t *restrict v = vmx;
        for (int x = x0; x < w; x++) {
            cD[x] = (uint16_t)(cD[x] + D[x]);
            cN[x] = (uint16_t)(cN[x] + (v[x] != 0));
        }
        if (y % rs == 0)
            for (int x = x0; x < w; x++) {
                const uint8_t d = c[x] > wa[x] ? (uint8_t)(c[x] - wa[x]) : (uint8_t)(wa[x] - c[x]);
                cS[x] = (uint16_t)(cS[x] + (uint8_t)(d & v[x]));
                cA[x] = (uint16_t)(cA[x] + (v[x] != 0));
            }
        if (y % cell == cell - 1 || y == h - 1) {          /* close this row of cells */
            const size_t ci = (size_t)(y / cell) * cw;
            for (int j = 0; j < cw; j++) {
                const int x0 = j * cell, x1 = mo_imin(x0 + cell, w);
                uint32_t sd = 0, nd = 0, sa = 0, na = 0;
                for (int x = x0; x < x1; x++) { sd += cD[x]; nd += cN[x]; sa += cS[x]; na += cA[x]; }
                p->SD[ci + j] = sd; p->ND[ci + j] = nd; p->SA[ci + j] = sa; p->NA[ci + j] = na;
            }
            memset(cD, 0, 4 * (size_t)w * sizeof(uint16_t));
        }
    }
}

TBOT_API int tbot_motion_kernel(tbot_motion_kparams *p) {
    const int h = p->h, w = p->w, cell = p->cell;
    const size_t n = (size_t)h * w;
    const int ch = (h + cell - 1) / cell, cw = (w + cell - 1) / cell;
    tbot_motion_warp(p->ref_a, p->src_valid, h, w, p->dxa, p->dya, p->block, p->wa, p->va);
    tbot_motion_warp(p->ref_b, p->src_valid, h, w, p->dxb, p->dyb, p->block, p->wb, p->vb);
    tbot_motion_gain(p->wa, (int64_t)n, p->ga, p->oa, p->wa);
    tbot_motion_gain(p->wb, (int64_t)n, p->gb, p->ob, p->wb);
    if (cell <= 256 && p->row_step >= 1 && p->env >= 1 && 6 * n >= 21 * (size_t)w + 32) {
        mo_kernel_rows(p, p->scratch);
    } else {
    tbot_motion_envelope(p->cur, p->wa, p->wb, p->va, p->vb, p->mask, h, w, p->env, p->scratch, p->D);
    uint8_t *valid = p->scratch + 5 * n;
    for (size_t i = 0; i < n; i++) valid[i] = (uint8_t)(p->va[i] & p->vb[i] & (p->mask ? p->mask[i] : 0xFF));
    tbot_motion_absdiff_sums(p->cur, p->wa, valid, h, w, cell, p->row_step, p->SA, p->NA);
    tbot_motion_cell_sums(p->D, h, w, cell, 1, p->SD);
    /* ND: count of valid != 0 (all rows), then np.maximum(ND, 1) */
    memset(p->ND, 0, (size_t)ch * cw * sizeof(int64_t));
    for (int y = 0; y < h; y++) {
        const uint8_t *restrict v = valid + (size_t)y * w;
        int64_t *o = p->ND + (size_t)(y / cell) * cw;
        for (int j = 0; j < cw; j++) {
            const int x0 = j * cell, x1 = mo_imin(x0 + cell, w);
            uint32_t c = 0;
            for (int x = x0; x < x1; x++) c += v[x] != 0;
            o[j] += c;
        }
    }
    }
    for (int k = 0; k < ch * cw; k++) if (p->ND[k] < 1) p->ND[k] = 1;
    p->sigma_n = tbot_motion_noise(p->SA, p->NA, ch * cw, cell, p->row_step, p->noise_quantile, p->noise_min_fill,
                                   p->abs_to_sigma);
    if (tbot_motion_threshold(p->SD, p->ND, p->SA, p->NA, ch, cw, p->sigma_n, p->t_min_dn, p->noise_k, p->cfar_gain,
                              p->texture_gain, p->abs_to_sigma, p->cfar_box, p->cfar_guard, p->thr, p->mD))
        return -1;
    tbot_motion_compare(p->D, h, w, p->thr, cw, cell, p->binary);
    return 0;
}

/* --------------------------------------------------------------------------------------- measurement */
typedef struct {
    int32_t h, w, cell, scale, min_area, max_area, max_cand, cap_runs;
    const uint8_t *binary, *D, *cur, *wa;
    const double *mD;
    uint8_t *joined;           /* h x w scratch */
    int32_t *runs;             /* cap_runs x 4 scratch */
    double *out;               /* max(max_cand, 0) x MO_DET_COLS */
    double S[9];               /* slot.S, row-major */
    double pixel_box_var, score_mass;
    int32_t n_labels, n_cand, n_out, roi[4];   /* out: labels incl. background (0: nothing set), candidates */
} tbot_motion_meas;

TBOT_API int tbot_motion_meas_size(void) { return (int)sizeof(tbot_motion_meas); }

typedef struct {
    int32_t key, label, area, bx0, by0, bx1, by1, qx, qy, pk;
    int64_t s0, sx, sy, sxx, syy, sxy;
} mo_comp;

static int mo_rank_cmp(const void *a, const void *b) {
    const mo_comp *x = *(const mo_comp *const *)a, *y = *(const mo_comp *const *)b;
    if (x->s0 != y->s0) return x->s0 > y->s0 ? -1 : 1;
    return (x->label > y->label) - (x->label < y->label);
}

/* Returns 0, or -1 (capacity / memory: the caller measures with OpenCV / NumPy) */
TBOT_API int tbot_motion_measure(tbot_motion_meas *p) {
    const int h = p->h, w = p->w;
    p->n_labels = p->n_cand = p->n_out = 0;
    p->roi[0] = p->roi[1] = p->roi[2] = p->roi[3] = 0;
    int minx = w, maxx = -1, miny = h, maxy = -1;
    for (int y = 0; y < h; y++) {
        const uint8_t *r = p->binary + (size_t)y * w;
        int x = 0;
        while (x + 8 <= w) {
            uint64_t v;
            memcpy(&v, r + x, 8);
            if (v) break;
            x += 8;
        }
        while (x < w && !r[x]) x++;
        if (x >= w) continue;
        if (x < minx) minx = x;
        int xe = w - 1;
        while (!r[xe]) xe--;
        if (xe > maxx) maxx = xe;
        if (y < miny) miny = y;
        maxy = y;
    }
    if (maxy < 0) return 0;
    p->roi[0] = minx; p->roi[1] = miny; p->roi[2] = maxx - minx + 1; p->roi[3] = maxy - miny + 1;
    const int X0 = mo_imax(0, minx - 1), Y0 = mo_imax(0, miny - 1);
    const int X1 = mo_imin(w, maxx + 2), Y1 = mo_imin(h, maxy + 2);
    const int rw = X1 - X0, rh = Y1 - Y0;
    /* joined = 3 x 3 max of the ROI (inside the ROI): columns first into a row, then rows */
    uint8_t *J = p->joined;
    for (int y = 0; y < rh; y++) {
        uint8_t *jr = J + (size_t)y * rw;
        const uint8_t *b0 = p->binary + (size_t)(Y0 + y) * w + X0;
        const uint8_t *bm = y > 0 ? b0 - w : NULL, *bp = y + 1 < rh ? b0 + w : NULL;
        for (int x = 0; x < rw; x++) jr[x] = (uint8_t)(b0[x] | (bm ? bm[x] : 0) | (bp ? bp[x] : 0));
    }
    /* horizontal pass in place with a carried left value */
    for (int y = 0; y < rh; y++) {
        uint8_t *jr = J + (size_t)y * rw;
        uint8_t left = 0;
        for (int x = 0; x < rw; x++) {
            const uint8_t c = jr[x], r = x + 1 < rw ? jr[x + 1] : 0;
            jr[x] = (uint8_t)(left | c | r);
            left = c;
        }
    }
    /* runs of the joined image (ROI coordinates) */
    int32_t nr = 0;
    int32_t *runs = p->runs;
    int32_t *row_start = (int32_t *)malloc((size_t)(rh + 1) * sizeof(int32_t));
    if (!row_start) return -1;
    for (int y = 0; y < rh; y++) {
        const uint8_t *jr = J + (size_t)y * rw;
        row_start[y] = nr;
        int x = 0;
        while (x < rw) {
            while (x < rw && !jr[x]) x++;
            if (x >= rw) break;
            const int x0 = x;
            while (x < rw && jr[x]) x++;
            if (nr >= p->cap_runs) { free(row_start); return -1; }
            int32_t *r = runs + 4 * (size_t)nr++;
            r[0] = y; r[1] = x0; r[2] = x; r[3] = 0;
        }
    }
    row_start[rh] = nr;
    int32_t *parent = (int32_t *)malloc((size_t)(nr > 0 ? nr : 1) * 2 * sizeof(int32_t));
    if (!parent) { free(row_start); return -1; }
    int32_t *slot = parent + nr;
    for (int32_t i = 0; i < nr; i++) parent[i] = i;
    for (int y = 1; y < rh; y++) {
        const int32_t p1 = row_start[y], c1 = row_start[y + 1];
        int32_t j = row_start[y - 1];
        for (int32_t i = p1; i < c1; i++) {
            const int32_t x0 = runs[4 * (size_t)i + 1], x1 = runs[4 * (size_t)i + 2];
            while (j < p1 && runs[4 * (size_t)j + 2] < x0) j++;
            for (int32_t k = j; k < p1 && runs[4 * (size_t)k + 1] <= x1; k++) {
                const int32_t a = front_find(parent, i), b = front_find(parent, k);
                if (a < b) parent[b] = a;
                else if (b < a) parent[a] = b;
            }
        }
    }
    int32_t ncomp = 0;
    for (int32_t i = 0; i < nr; i++) ncomp += parent[i] == i;
    mo_comp *cs = (mo_comp *)calloc((size_t)(ncomp > 0 ? ncomp : 1), sizeof(mo_comp));
    front_order *ord = (front_order *)malloc((size_t)(ncomp > 0 ? ncomp : 1) * sizeof(front_order));
    mo_comp **rank = (mo_comp **)malloc((size_t)(ncomp > 0 ? ncomp : 1) * sizeof(mo_comp *));
    if (!cs || !ord || !rank) { free(row_start); free(parent); free(cs); free(ord); free(rank); return -1; }
    const int32_t bw2 = (rw + 1) / 2;
    int32_t nc = 0;
    for (int32_t i = 0; i < nr; i++) {
        const int32_t *r = runs + 4 * (size_t)i;
        const int32_t root = front_find(parent, i);
        const int32_t key = (r[0] >> 1) * bw2 + (r[1] >> 1);
        mo_comp *c;
        if (root == i) {
            slot[i] = nc;
            c = cs + nc++;
            c->key = key; c->bx0 = c->by0 = 1 << 30; c->bx1 = c->by1 = -1; c->pk = -1;
        } else {
            c = cs + slot[root];
            if (key < c->key) c->key = key;
        }
        /* the thresholded pixels of the run (raster order), global coordinates */
        const int gy = Y0 + r[0];
        const uint8_t *br = p->binary + (size_t)gy * w, *dr = p->D + (size_t)gy * w;
        for (int x = r[1]; x < r[2]; x++) {
            const int gx = X0 + x;
            if (!br[gx]) continue;
            const int64_t d = dr[gx];
            c->area++;
            c->s0 += d; c->sx += d * gx; c->sy += d * gy;
            c->sxx += d * gx * gx; c->syy += d * gy * gy; c->sxy += d * gx * gy;
            if (gx < c->bx0) c->bx0 = gx;
            if (gx > c->bx1) c->bx1 = gx;
            if (gy < c->by0) c->by0 = gy;
            if (gy > c->by1) c->by1 = gy;
            if ((int32_t)d > c->pk) { c->pk = (int32_t)d; c->qx = gx; c->qy = gy; }
        }
    }
    for (int32_t k = 0; k < nc; k++) { ord[k].key = cs[k].key; ord[k].root = k; }
    qsort(ord, (size_t)nc, sizeof(front_order), front_order_cmp);
    for (int32_t k = 0; k < nc; k++) cs[ord[k].root].label = k + 1;
    p->n_labels = nc + 1;
    int32_t nk = 0;
    for (int32_t k = 0; k < nc; k++)
        if (cs[k].area >= p->min_area && cs[k].area <= p->max_area) rank[nk++] = cs + k;
    p->n_cand = nk;
    qsort(rank, (size_t)nk, sizeof(mo_comp *), mo_rank_cmp);
    const int mc = p->max_cand;
    int nout = mc >= 0 ? mo_imin(mc, nk) : mo_imax(nk + mc, 0);      /* rank[:max_candidates] */
    const double s = (double)p->scale, f_area = (double)(p->scale * p->scale);
    const int cw = (w + p->cell - 1) / p->cell;
    const double *S = p->S;
    int m = 0;
    for (int t = 0; t < nout; t++) {
        const mo_comp *c = rank[t];
        const double m0 = (double)c->s0;
        if (m0 <= 0.0) continue;
        const double cx = (double)c->sx / m0, cy = (double)c->sy / m0;
        const double vxx = (double)c->sxx / m0 - cx * cx - p->pixel_box_var * 1.0;
        const double vyy = (double)c->syy / m0 - cy * cy - p->pixel_box_var * 1.0;
        const double vxy = (double)c->sxy / m0 - cx * cy;
        const double tr = vxx + vyy, det = vxx * vyy - vxy * vxy;
        const double disc = sqrt(mo_pymax(tr * tr * 0.25 - det, 0.0));
        const double lmax = tr * 0.5 + disc, lmin = mo_pymax(tr * 0.5 - disc, 0.0);
        const double diam = mo_pymax(1.0, 4.0 * sqrt(mo_pymax(lmax, 0.0)) * s);
        const double pkv = (double)c->pk;
        const double contrast = (double)((int)p->cur[(size_t)c->qy * w + c->qx] - (int)p->wa[(size_t)c->qy * w + c->qx]);
        const double local = p->mD[(size_t)(c->qy / p->cell) * cw + c->qx / p->cell];
        const double u = (cx + 0.5) * s - 0.5, v = (cy + 0.5) * s - 0.5;
        const double ww = S[6] * u + S[7] * v + S[8];
        const double st_u = (S[0] * u + S[1] * v + S[2]) / ww, st_v = (S[3] * u + S[4] * v + S[5]) / ww;
        const double mass = m0 * f_area;
        double *o = p->out + (size_t)m++ * MO_DET_COLS;
        o[0] = u; o[1] = v; o[2] = (double)c->area * f_area; o[3] = diam; o[4] = contrast;
        o[5] = pkv / mo_pymax(local, 1.0); o[6] = mass / (mass + p->score_mass);
        o[7] = (double)c->bx0 * s; o[8] = (double)c->by0 * s;
        o[9] = (double)(c->bx1 - c->bx0 + 1) * s; o[10] = (double)(c->by1 - c->by0 + 1) * s;
        o[11] = st_u; o[12] = st_v; o[13] = mass; o[14] = mo_pymax(1.0, 4.0 * sqrt(lmin) * s);
    }
    p->n_out = m;
    free(row_start); free(parent); free(cs); free(ord); free(rank);
    return 0;
}
