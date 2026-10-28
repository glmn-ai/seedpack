"""Deterministic arithmetic for the coders' decisions: the same bits on the GPU and with --cpu.

The side information a coder stores (scale bins, column classes, table choices, row parents) is
decided from sums, logarithms and small least-squares fits. CuPy and NumPy sum in different
orders, round log2 / exp2 differently and use different BLAS / LAPACK, so a decision near a bin
edge can come out differently on the two, and the archives differ (both decodable: the decoder
reads the stored side information). These kernels make every such number a fixed function of
its inputs: one thread per output summing in index order (float64), log2 / exp2 from IEEE basic
operations only, small systems by Gaussian elimination, and similarities of int8 rows as float32
products that are exact in any order (sim8). The module is compiled with --fmad=false on the GPU
and without contraction on the CPU (cpukernels).
"""
import numpy as np

from .xp import cp

KERNELS = r"""
typedef unsigned char uchar;

__device__ __forceinline__ double dm_rint(double x) {        // half to even: the addition rounds
    if (!(fabs(x) < 4503599627370496.0)) return x;
    double y = fabs(x) + 4503599627370496.0;
    y = y - 4503599627370496.0;
    return copysign(y, x);
}

__device__ double dm_log2v(double x) {
    if (isnan(x)) return x;
    if (x < 0.0) return __longlong_as_double(0x7ff8000000000000LL);
    if (x == 0.0) return __longlong_as_double((long long)0xfff0000000000000ULL);
    if (isinf(x)) return x;
    long long u = __double_as_longlong(x);
    int e = (int)((u >> 52) & 0x7ff);
    if (e == 0) {                                               // subnormal: scaled by 2^54
        x = x * 18014398509481984.0;
        u = __double_as_longlong(x);
        e = (int)((u >> 52) & 0x7ff) - 54;
    }
    e -= 1023;
    double m = __longlong_as_double((u & 0x000fffffffffffffLL) | 0x3ff0000000000000LL);
    if (m > 1.4142135623730951) { m = m * 0.5; e += 1; }
    double s = (m - 1.0) / (m + 1.0), z = s * s;                // ln m = 2 atanh s, |s| < 0.172
    double p = 0.04;
    p = p * z + 0.043478260869565216; p = p * z + 0.047619047619047616; p = p * z + 0.05263157894736842;
    p = p * z + 0.058823529411764705; p = p * z + 0.06666666666666667; p = p * z + 0.07692307692307693;
    p = p * z + 0.09090909090909091;  p = p * z + 0.1111111111111111;  p = p * z + 0.14285714285714285;
    p = p * z + 0.2; p = p * z + 0.3333333333333333; p = p * z + 1.0;
    return (double)e + (2.0 * s * p) * 1.4426950408889634;
}

__device__ double dm_exp2v(double y) {
    if (isnan(y)) return y;
    if (y >= 1024.0) return __longlong_as_double(0x7ff0000000000000LL);
    if (y < -1075.0) return 0.0;
    double n = dm_rint(y), t = (y - n) * 0.6931471805599453;  // e^t, |t| <= 0.347
    double p = 2.8114572543455206e-15;
    p = p * t + 4.779477332387385e-14;  p = p * t + 7.647163731819816e-13;  p = p * t + 1.1470745597729725e-11;
    p = p * t + 1.6059043836821613e-10; p = p * t + 2.08767569878681e-09;   p = p * t + 2.505210838544172e-08;
    p = p * t + 2.7557319223985893e-07; p = p * t + 2.755731922398589e-06;  p = p * t + 2.48015873015873e-05;
    p = p * t + 0.0001984126984126984;  p = p * t + 0.001388888888888889;   p = p * t + 0.008333333333333333;
    p = p * t + 0.041666666666666664;   p = p * t + 0.16666666666666666;    p = p * t + 0.5;
    p = p * t + 1.0; p = p * t + 1.0;
    int k = (int)n, k1 = k / 2, k2 = k - k1;                   // 2^k in two normal factors
    return p * __longlong_as_double((long long)(k1 + 1023) << 52) * __longlong_as_double((long long)(k2 + 1023) << 52);
}

extern "C" __global__ void dm_log2(const double* x, long long n, double* y) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = dm_log2v(x[i]);
}
extern "C" __global__ void dm_exp2(const double* x, long long n, double* y) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = dm_exp2v(x[i]);
}

// sums in index order, float64; sq: of squares
extern "C" __global__ void dm_rows(const float* x, long long nr, long long nc, int sq, double* out) {
    long long r = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= nr) return;
    const float* p = x + r * nc;
    double s = 0.0;
    if (sq) for (long long j = 0; j < nc; j++) { double v = p[j]; s = s + v * v; }
    else for (long long j = 0; j < nc; j++) s = s + (double)p[j];
    out[r] = s;
}
extern "C" __global__ void dm_rows_d(const double* x, long long nr, long long nc, int sq, double* out) {
    long long r = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= nr) return;
    const double* p = x + r * nc;
    double s = 0.0;
    if (sq) for (long long j = 0; j < nc; j++) s = s + p[j] * p[j];
    else for (long long j = 0; j < nc; j++) s = s + p[j];
    out[r] = s;
}
extern "C" __global__ void dm_cols(const float* x, long long nr, long long nc, int sq, const uchar* mask, int masked,
                                   double* out) {
    long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nc) return;
    double s = 0.0;
    for (long long r = 0; r < nr; r++) {
        if (masked && !mask[r]) continue;
        double v = x[r * nc + c];
        s = s + (sq ? v * v : v);
    }
    out[c] = s;
}

// batched: G[c] = X[c] X[c]^T (X: nb x k x d), X[c] y[c]; float64 sums, float32 out
extern "C" __global__ void dm_gram(const float* X, long long nb, int k, long long d, float* out) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nb * k * k) return;
    long long c = i / (k * k); int a = (int)((i / k) % k), b = (int)(i % k);
    const float* xa = X + (c * k + a) * d; const float* xb = X + (c * k + b) * d;
    double s = 0.0;
    for (long long t = 0; t < d; t++) s = s + (double)xa[t] * (double)xb[t];
    out[i] = (float)s;
}
extern "C" __global__ void dm_bdot(const float* X, const float* y, long long nb, int k, long long d, float* out) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nb * k) return;
    long long c = i / k;
    const float* xa = X + i * d; const float* yc = y + c * d;
    double s = 0.0;
    for (long long t = 0; t < d; t++) s = s + (double)xa[t] * (double)yc[t];
    out[i] = (float)s;
}
// batched small systems A[c] x = b[c] (k <= 16): Gaussian elimination with partial pivoting, float64
extern "C" __global__ void dm_solve(const float* A, const float* b, long long nb, int k, float* x) {
    long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nb) return;
    double M[16][17];
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) M[i][j] = A[(c * k + i) * k + j];
        M[i][k] = b[c * k + i];
    }
    for (int col = 0; col < k; col++) {
        int piv = col;
        for (int i = col + 1; i < k; i++) if (fabs(M[i][col]) > fabs(M[piv][col])) piv = i;
        if (piv != col) for (int j = 0; j <= k; j++) { double t = M[col][j]; M[col][j] = M[piv][j]; M[piv][j] = t; }
        double p = M[col][col];
        for (int i = col + 1; i < k; i++) {
            double f = M[i][col] / p;
            for (int j = col; j <= k; j++) M[i][j] = M[i][j] - f * M[col][j];
        }
    }
    for (int i = k - 1; i >= 0; i--) {
        double s = M[i][k];
        for (int j = i + 1; j < k; j++) s = s - M[i][j] * M[j][k];
        M[i][k] = s / M[i][i];
    }
    for (int i = 0; i < k; i++) x[c * k + i] = (float)M[i][k];
}
"""

_mod = None


def _k(name):
    global _mod
    if _mod is None:
        _mod = cp.RawModule(code=KERNELS, options=("-std=c++17", "--fmad=false"))
    return _mod.get_function(name)


def _launch(name, n, *args):
    if n > 0:
        _k(name)(((n + 255) // 256,), (256,), args)


def _f64(x):
    return cp.ascontiguousarray(cp.asarray(x), dtype=cp.float64)


def _f32(x):
    return cp.ascontiguousarray(cp.asarray(x), dtype=cp.float32)


def log2(x):
    """float64 log2 of x, elementwise"""
    x = _f64(x)
    y = cp.empty(x.shape, cp.float64)
    _launch("dm_log2", x.size, x, np.int64(x.size), y)
    return y


def exp2(x):
    """float64 2 ** x, elementwise"""
    x = _f64(x)
    y = cp.empty(x.shape, cp.float64)
    _launch("dm_exp2", x.size, x, np.int64(x.size), y)
    return y


def rows(x, square=False):
    """float64 sums (of squares) along the rows of a 2-D array"""
    x = cp.asarray(x)
    x = cp.ascontiguousarray(x) if x.dtype == cp.float64 else _f32(x)
    nr, nc = x.shape
    out = cp.empty(nr, cp.float64)
    _launch("dm_rows_d" if x.dtype == cp.float64 else "dm_rows", nr, x, np.int64(nr), np.int64(nc), np.int32(square), out)
    return out


def cols(x, square=False, mask=None):
    """float64 sums (of squares) down the columns of a 2-D float32 array, rows with mask False left out"""
    x = _f32(x)
    nr, nc = x.shape
    m = cp.ascontiguousarray(cp.asarray(mask), dtype=cp.uint8) if mask is not None else cp.zeros(1, cp.uint8)
    out = cp.empty(nc, cp.float64)
    _launch("dm_cols", nc, x, np.int64(nr), np.int64(nc), np.int32(square), m, np.int32(mask is not None), out)
    return out


def total(x, square=False):
    """float64 sum (of squares) of all elements, in index order"""
    return rows(cp.asarray(x).reshape(1, -1), square)[0]


def gram(X):
    """X (n, k, d) float32 -> X X^T per batch (n, k, k) float32"""
    X = _f32(X)
    n, k, d = X.shape
    out = cp.empty((n, k, k), cp.float32)
    _launch("dm_gram", n * k * k, X, np.int64(n), np.int32(k), np.int64(d), out)
    return out


def bdot(X, y):
    """X (n, k, d), y (n, d) float32 -> X y per batch (n, k) float32"""
    X, y = _f32(X), _f32(y)
    n, k, d = X.shape
    out = cp.empty((n, k), cp.float32)
    _launch("dm_bdot", n * k, X, y, np.int64(n), np.int32(k), np.int64(d), out)
    return out


def solve(A, b):
    """batched small systems A (n, k, k) x = b (n, k), k <= 16 -> x (n, k) float32"""
    A, b = _f32(A), _f32(b)
    n, k = b.shape
    assert k <= 16
    x = cp.empty((n, k), cp.float32)
    _launch("dm_solve", n, A, b, np.int64(n), np.int32(k), x)
    return x


SIM_COLS = 1024        # columns per float32 product in sim8: |a sum| <= 1024 * 127^2 < 2^24


def sim8(A, B):
    """int8 rows A (ra, C), B (rb, C) with values in -127..127 -> exact dot products (ra, rb) int32.
    A float32 matrix product over at most SIM_COLS columns is exact whatever the BLAS's order (every
    partial sum is an integer below 2^24), so cuBLAS and the CPU's BLAS give the same, quickly"""
    A, B = cp.asarray(A), cp.asarray(B)
    S = None
    for c0 in range(0, A.shape[1], SIM_COLS):
        p = (A[:, c0:c0 + SIM_COLS].astype(cp.float32) @ B[:, c0:c0 + SIM_COLS].astype(cp.float32).T).astype(cp.int32)
        S = p if S is None else S + p
    return S
