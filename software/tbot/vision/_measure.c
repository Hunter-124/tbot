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
 * Build: scripts/build_native.sh (cc -O2 -shared -fPIC -ffp-contract=off). Change TBOT_MEASURE_ABI here and
 * in _native.py together whenever a signature or struct changes: a stale library is then refused.
 */
#include <math.h>
#include <stdint.h>
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
