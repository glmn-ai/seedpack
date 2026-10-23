"""QR: quantizer-aware residual coding of quantised tensors predicted from a seed.

Re-quantising the seed (repo3) predicts a file's blocks; XOR + zstd then codes the differences.
For the quant types with per-block scales and round-to-nearest codes (legacy Q4/Q5/Q8, K-quants,
IQ4) a much better model exists: llama.cpp first chooses a block's scales (this is where the
importance matrix acts) and then sets every code to the nearest grid point,
code = round((x - c) / a), with the final scales a, c. So:

  * the scale bytes are coded first, as XOR with the predicted block's (context: byte position);
  * then, with the true scales known, each code is predicted as round(u), u = (x - c) / a with x
    the seed's value of that weight, and the difference is coded with the context
    (seed uncertainty in code steps, distance of u to the rounding boundary).

Weights far from a boundary cost almost nothing, so the residual shrinks 2-4x and the coding runs
on the GPU (interleaved rANS segments, as M21) instead of zstd on the CPU. The model only has to
be deterministic, not exact: encoder and decoder run the same kernels, and the actual codes are
coded losslessly whatever the prediction.

Seeds: F32 / F16 / BF16 (exact values) and any QR type (dequantised here; uncertainty = half a
code step). Tensors whose seed is another type stay with XOR.
"""
import json
import struct

import cupy as cp
import numpy as np

from .codecs_gguf import pack, unpack, zstd_c, zstd_d
from .gpu2 import cum_of, store_tables, load_tables

QR_TYPES = {2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 20, 23}
SEED_TYPES = QR_TYPES | {0, 1, 30}
EXPLICIT = 99                            # seed given as float32 (x, h) pairs per weight
BS = {2: 18, 3: 20, 6: 22, 7: 24, 8: 34, 10: 84, 11: 110, 12: 144, 13: 176, 14: 210, 20: 18, 23: 136}
QK = {t: (32 if t in (2, 3, 6, 7, 8, 20) else 256) for t in QR_TYPES}
NFIELDS = {2: 1, 6: 1, 8: 1, 20: 1, 3: 2, 7: 2, 10: 34, 11: 17, 14: 17, 12: 18, 13: 18, 23: 9}
NSB_MAX = 20
NTAB = NSB_MAX + 64 + 2                  # scale byte positions, (uncertainty x margin) code contexts, block flag
PB = 12
BATCH = 384 << 20                        # target bytes per coding batch (bounds GPU memory)
STATS = None                             # dict -> per (type, stream) entropy in bytes (research)

KERNELS = r"""
#define PB 12
#define PSCALE (1u << PB)
#define RANS_L (1u << 23)
#define NSB_MAX 20
#define INFINITY __int_as_float(0x7f800000)
#define HBOFF_COL 3
#define NTAB (NSB_MAX + 64 + 2)
#define FLAG (NSB_MAX + 64)                // + previous block's flag: 1 = block equals its prediction
typedef unsigned char uchar;

__constant__ float KV[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

__device__ __forceinline__ float h2f(const uchar* p) {
    unsigned int h = p[0] | (p[1] << 8);
    unsigned int s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023, f;
    if (e == 0) {
        if (m == 0) f = s << 31;
        else { int ee = 113; while (!(m & 1024)) { m <<= 1; ee--; } m &= 1023; f = (s << 31) | (ee << 23) | (m << 13); }
    } else if (e == 31) f = (s << 31) | 0x7f800000u | (m << 13);
    else f = (s << 31) | ((e + 112) << 23) | (m << 13);
    return __int_as_float(f);
}

__device__ __forceinline__ int QKof(int T) { return (T == 2 || T == 3 || T == 6 || T == 7 || T == 8 || T == 20) ? 32 : 256; }
__device__ __forceinline__ int BSof(int T) {
    switch (T) { case 2: return 18; case 3: return 20; case 6: return 22; case 7: return 24; case 8: return 34;
                 case 10: return 84; case 11: return 110; case 12: return 144; case 13: return 176; case 14: return 210;
                 case 20: return 18; case 23: return 136; }
    return 1;
}
__device__ __forceinline__ int NSB(int T) {
    switch (T) { case 2: case 6: case 8: case 20: return 2; case 3: case 7: return 4; case 10: return 20;
                 case 11: return 14; case 12: case 13: return 16; case 14: return 18; case 23: return 8; }
    return 0;
}
__device__ __forceinline__ int SBO(int T, int k) {       // byte offset of the k-th scale byte
    switch (T) { case 10: return k < 16 ? k : 80 + (k - 16); case 11: return 96 + k; case 14: return 192 + k; }
    return k;
}
__device__ __forceinline__ int QMIN(int T) { return T == 8 ? -128 : 0; }
__device__ __forceinline__ int QMAX(int T) {
    switch (T) { case 10: return 3; case 11: return 7; case 6: case 7: case 13: return 31; case 14: return 63; case 8: return 127; }
    return 15;
}
__device__ __forceinline__ bool NONLIN(int T) { return T == 20 || T == 23; }
__device__ __forceinline__ int same(const uchar* a, const uchar* b, int n) {
    for (int i = 0; i < n; i++) if (a[i] != b[i]) return 0;
    return 1;
}

__device__ __forceinline__ void scale_min_k4(int j, const uchar* q, int& sc, int& m) {
    if (j < 4) { sc = q[j] & 63; m = q[j + 4] & 63; }
    else { sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}
__device__ __forceinline__ int q3k_scale(const uchar* s, int is) {
    unsigned int a0 = s[0] | (s[1] << 8) | (s[2] << 16) | ((unsigned)s[3] << 24);
    unsigned int a1 = s[4] | (s[5] << 8) | (s[6] << 16) | ((unsigned)s[7] << 24);
    unsigned int tmp = s[8] | (s[9] << 8) | (s[10] << 16) | ((unsigned)s[11] << 24);
    const unsigned int k1 = 0x03030303u, k2 = 0x0f0f0f0fu;
    unsigned int r[4];
    r[2] = ((a0 >> 4) & k2) | (((tmp >> 4) & k1) << 4);
    r[3] = ((a1 >> 4) & k2) | (((tmp >> 6) & k1) << 4);
    r[0] = (a0 & k2) | (((tmp >> 0) & k1) << 4);
    r[1] = (a1 & k2) | (((tmp >> 2) & k1) << 4);
    return (int)(signed char)((r[is >> 2] >> (8 * (is & 3))) & 0xff);
}

__device__ int get_code(int T, const uchar* b, int w) {
    switch (T) {
    case 2: case 20: return w < 16 ? (b[2 + w] & 15) : (b[2 + w - 16] >> 4);
    case 3: return w < 16 ? (b[4 + w] & 15) : (b[4 + w - 16] >> 4);
    case 6: { int lo = w < 16 ? (b[6 + w] & 15) : (b[6 + w - 16] >> 4); return lo | (((b[2 + (w >> 3)] >> (w & 7)) & 1) << 4); }
    case 7: { int lo = w < 16 ? (b[8 + w] & 15) : (b[8 + w - 16] >> 4); return lo | (((b[4 + (w >> 3)] >> (w & 7)) & 1) << 4); }
    case 8: return (int)(signed char)b[2 + w];
    case 10: { int n = w >> 7, r = w & 127, j = r >> 5, l = r & 31; return (b[16 + 32 * n + l] >> (2 * j)) & 3; }
    case 11: { int n = w >> 7, r = w & 127, j = r >> 5, l = r & 31;
               return ((b[32 + 32 * n + l] >> (2 * j)) & 3) | (((b[l] >> (4 * n + j)) & 1) << 2); }
    case 12: { int c = w >> 6, hf = (w >> 5) & 1, l = w & 31; int v = b[16 + 32 * c + l]; return hf ? v >> 4 : v & 15; }
    case 13: { int c = w >> 6, hf = (w >> 5) & 1, l = w & 31; int v = b[48 + 32 * c + l];
               return (hf ? v >> 4 : v & 15) | (((b[16 + l] >> (2 * c + hf)) & 1) << 4); }
    case 14: { int n = w >> 7, r = w & 127, g = r >> 5, l = r & 31; int v = b[64 * n + l + ((g & 1) ? 32 : 0)];
               return ((g < 2) ? (v & 15) : (v >> 4)) | (((b[128 + 32 * n + l] >> (2 * g)) & 3) << 4); }
    case 23: { int ib = w >> 5, l = w & 31; int v = b[8 + 16 * ib + (l & 15)]; return l < 16 ? (v & 15) : (v >> 4); }
    }
    return 0;
}

// code bytes must be zero before (bits are OR-ed in)
__device__ void put_code(int T, uchar* b, int w, int q) {
    switch (T) {
    case 2: case 20: if (w < 16) b[2 + w] |= q; else b[2 + w - 16] |= q << 4; return;
    case 3: if (w < 16) b[4 + w] |= q; else b[4 + w - 16] |= q << 4; return;
    case 6: if (w < 16) b[6 + w] |= q & 15; else b[6 + w - 16] |= (q & 15) << 4; b[2 + (w >> 3)] |= ((q >> 4) & 1) << (w & 7); return;
    case 7: if (w < 16) b[8 + w] |= q & 15; else b[8 + w - 16] |= (q & 15) << 4; b[4 + (w >> 3)] |= ((q >> 4) & 1) << (w & 7); return;
    case 8: b[2 + w] = (uchar)q; return;
    case 10: { int n = w >> 7, r = w & 127, j = r >> 5, l = r & 31; b[16 + 32 * n + l] |= (q & 3) << (2 * j); return; }
    case 11: { int n = w >> 7, r = w & 127, j = r >> 5, l = r & 31;
               b[32 + 32 * n + l] |= (q & 3) << (2 * j); b[l] |= ((q >> 2) & 1) << (4 * n + j); return; }
    case 12: { int c = w >> 6, hf = (w >> 5) & 1, l = w & 31; b[16 + 32 * c + l] |= hf ? q << 4 : q; return; }
    case 13: { int c = w >> 6, hf = (w >> 5) & 1, l = w & 31; b[48 + 32 * c + l] |= hf ? (q & 15) << 4 : (q & 15);
               b[16 + l] |= ((q >> 4) & 1) << (2 * c + hf); return; }
    case 14: { int n = w >> 7, r = w & 127, g = r >> 5, l = r & 31;
               b[64 * n + l + ((g & 1) ? 32 : 0)] |= (g < 2) ? (q & 15) : ((q & 15) << 4);
               b[128 + 32 * n + l] |= ((q >> 4) & 3) << (2 * g); return; }
    case 23: { int ib = w >> 5, l = w & 31; b[8 + 16 * ib + (l & 15)] |= l < 16 ? q : q << 4; return; }
    }
}

// ---- scale fields: fp16 d / dmin / m and the 4-, 6-, 8-bit sub-scales, as integers
#define MAXF 34
__device__ __forceinline__ int NFIELD(int T) {
    switch (T) { case 2: case 6: case 8: case 20: return 1; case 3: case 7: return 2; case 10: return 34;
                 case 11: case 14: return 17; case 12: case 13: return 18; case 23: return 9; }
    return 0;
}
// 0 = fp16 scale, 1 = fp16 min, 2 = sub-scale, 3 = sub-min
__device__ __forceinline__ int FKIND(int T, int k) {
    switch (T) {
    case 3: case 7: return k == 0 ? 0 : 1;
    case 10: return k == 0 ? 0 : (k == 1 ? 1 : (k < 18 ? 2 : 3));
    case 12: case 13: return k == 0 ? 0 : (k == 1 ? 1 : (k < 10 ? 2 : 3));
    case 11: case 14: case 23: return k == 0 ? 0 : 2;
    }
    return 0;
}
__device__ __forceinline__ int u16(const uchar* b) { return b[0] | (b[1] << 8); }
__device__ int get_field(int T, const uchar* b, int k) {
    switch (T) {
    case 2: case 6: case 8: case 20: return u16(b);
    case 3: case 7: return u16(b + 2 * k);
    case 10: if (k == 0) return u16(b + 80); if (k == 1) return u16(b + 82); return k < 18 ? (b[k - 2] & 15) : (b[k - 18] >> 4);
    case 11: if (k == 0) return u16(b + 108); return q3k_scale(b + 96, k - 1);
    case 12: case 13: { if (k == 0) return u16(b); if (k == 1) return u16(b + 2);
                        int sc, m; scale_min_k4(k < 10 ? k - 2 : k - 10, b + 4, sc, m); return k < 10 ? sc : m; }
    case 14: if (k == 0) return u16(b + 208); return (int)(signed char)b[192 + k - 1];
    case 23: { if (k == 0) return u16(b); int ib = k - 1, sh = u16(b + 2);
               return ((b[4 + ib / 2] >> (4 * (ib % 2))) & 15) | (((sh >> (2 * ib)) & 3) << 4); }
    }
    return 0;
}
__device__ __forceinline__ void put16(uchar* b, int v) { b[0] = (uchar)(v & 255); b[1] = (uchar)((v >> 8) & 255); }
// writes all scale fields (the scale bytes must be zero before)
__device__ void set_fields(int T, uchar* b, const int* v) {
    switch (T) {
    case 2: case 6: case 8: case 20: put16(b, v[0]); return;
    case 3: case 7: put16(b, v[0]); put16(b + 2, v[1]); return;
    case 10: put16(b + 80, v[0]); put16(b + 82, v[1]);
             for (int j = 0; j < 16; j++) b[j] = (uchar)((v[2 + j] & 15) | ((v[18 + j] & 15) << 4)); return;
    case 11: put16(b + 108, v[0]);
             for (int j = 0; j < 16; j++) { int l = v[1 + j];
                 if (j < 8) b[96 + j] |= l & 15; else b[96 + j - 8] |= (l & 15) << 4;
                 b[96 + 8 + (j & 3)] |= ((l >> 4) & 3) << (2 * (j >> 2)); }
             return;
    case 12: case 13: put16(b, v[0]); put16(b + 2, v[1]);
             for (int j = 0; j < 4; j++) {
                 b[4 + j] = (uchar)(v[2 + j] | ((v[2 + j + 4] >> 4) << 6));
                 b[8 + j] = (uchar)(v[10 + j] | ((v[10 + j + 4] >> 4) << 6));
                 b[12 + j] = (uchar)((v[2 + j + 4] & 15) | ((v[10 + j + 4] & 15) << 4)); }
             return;
    case 14: put16(b + 208, v[0]); for (int j = 0; j < 16; j++) b[192 + j] = (uchar)v[1 + j]; return;
    case 23: { put16(b, v[0]); int sh = 0;
               for (int ib = 0; ib < 8; ib++) { int ls = v[1 + ib];
                   b[4 + ib / 2] |= (ls & 15) << (4 * (ib % 2)); sh |= ((ls >> 4) & 3) << (2 * ib); }
               put16(b + 2, sh); return; }
    }
}
// field delta -> symbol(s): small fields mod 256; fp16 fields: |diff| <= 127 in one symbol, else
// escape (-128) + the two raw bytes
#define FCTX(kind, any) ((kind) * 2 + (any))
#define FESC_LO 8
#define FESC_HI 9

// value = a * code + c (NONLIN types: value = a * KV[code], c unused)
__device__ void wscale(int T, const uchar* b, int w, float& a, float& c) {
    c = 0.f;
    switch (T) {
    case 2: a = h2f(b); c = __fmul_rn(-8.f, a); return;
    case 3: case 7: a = h2f(b); c = h2f(b + 2); return;
    case 6: a = h2f(b); c = __fmul_rn(-16.f, a); return;
    case 8: case 20: a = h2f(b); return;
    case 10: { int n = w >> 7, r = w & 127, j = r >> 5; int sc = b[8 * n + 2 * j + ((r & 31) >= 16)];
               a = __fmul_rn(h2f(b + 80), (float)(sc & 15)); c = -__fmul_rn(h2f(b + 82), (float)(sc >> 4)); return; }
    case 11: { int n = w >> 7, r = w & 127, j = r >> 5; int s = q3k_scale(b + 96, 8 * n + 2 * j + ((r & 31) >= 16));
               a = __fmul_rn(h2f(b + 108), (float)(s - 32)); c = __fmul_rn(-4.f, a); return; }
    case 12: case 13: { int sc, m; scale_min_k4(w >> 5, b + 4, sc, m);
               a = __fmul_rn(h2f(b), (float)sc); c = -__fmul_rn(h2f(b + 2), (float)m); return; }
    case 14: { int n = w >> 7, r = w & 127, g = r >> 5, l = r & 31;
               a = __fmul_rn(h2f(b + 208), (float)(signed char)b[192 + 8 * n + (l >> 4) + 2 * g]); c = __fmul_rn(-32.f, a); return; }
    case 23: { int ib = w >> 5; int sh = b[2] | (b[3] << 8);
               int ls = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 15) | (((sh >> (2 * ib)) & 3) << 4);
               a = __fmul_rn(h2f(b), (float)(ls - 32)); return; }
    }
    a = 0.f;
}

// weights sharing one (a, c): the scale computations are cached per such run
__device__ __forceinline__ int SUBLEN(int T) { return (T == 10 || T == 11 || T == 14) ? 16 : 32; }
struct SC { long long key; float a, c; };

// the seed's value of global weight gw and its half uncertainty
__device__ void seed_val(int ST, const uchar* sb, long long gw, float& x, float& h, SC& cache, int cols) {
    h = 0.f;
    if (ST == 98) { x = 0.f; h = ((const float*)sb)[gw % cols]; return; }    // no seed: per-column RMS
    if (ST == 99) { const float* f = (const float*)sb; x = f[2 * gw]; h = f[2 * gw + 1]; return; }   // explicit (x, h)
    if (ST == 0) { x = ((const float*)sb)[gw]; return; }
    if (ST == 1) { x = h2f(sb + 2 * gw); return; }
    if (ST == 30) { x = __int_as_float(((unsigned int)(sb[2 * gw] | (sb[2 * gw + 1] << 8))) << 16); return; }
    int qk = QKof(ST);
    const uchar* blk = sb + (gw / qk) * BSof(ST);
    int w = (int)(gw % qk), q = get_code(ST, blk, w);
    long long key = gw / SUBLEN(ST);
    if (key != cache.key) { cache.key = key; wscale(ST, blk, w, cache.a, cache.c); }
    float a = cache.a, c = cache.c;
    if (NONLIN(ST)) {
        x = __fmul_rn(a, KV[q]);
        float g = q < 15 ? KV[q + 1] - KV[q] : KV[15] - KV[14];
        h = __fmul_rn(fabsf(a), 0.5f * g);
    } else {
        x = __fadd_rn(__fmul_rn(a, (float)q), c);
        h = __fmul_rn(fabsf(a), 0.5f);
    }
}

// predicted code r, mirror sign s and context of target weight w, from the TRUE block scales
__device__ void predict_code(int T, float a, float c, const uchar* pblk, int w, float x, float h, int hboff, int& r, int& s, int& ctx) {
    float u, hu;
    if (NONLIN(T)) {
        float t = __fdiv_rn(x, a);
        int i = 0;
        while (i < 14 && t >= KV[i + 1]) i++;
        float g = KV[i + 1] - KV[i];
        u = __fadd_rn((float)i, __fdiv_rn(__fsub_rn(t, KV[i]), g));
        hu = __fdiv_rn(h, __fmul_rn(fabsf(a), g));
    } else {
        u = __fdiv_rn(__fsub_rn(x, c), a);
        hu = __fdiv_rn(h, fabsf(a));
    }
    if (!isfinite(u) || !isfinite(hu)) {                      // zero scale: use the predicted block's code
        r = get_code(T, pblk, w); s = 1; ctx = NSB_MAX + 63; return;
    }
    float ru = rintf(u), fr = __fsub_rn(u, ru);
    int q = (int)fmaxf(fminf(ru, (float)QMAX(T)), (float)QMIN(T));
    r = q; s = fr >= 0.f ? 1 : -1;
    int mb = min(7, (int)(fabsf(fr) * 16.f));
    if (hboff == HBOFF_COL) {            // no seed: r is the zero point; its place in the code range matters
        s = 1;
        mb = min(7, max(0, (q - QMIN(T)) * 8 / (QMAX(T) - QMIN(T) + 1)));
    }
    int hb = 0;
    if (hu > 0.f) { int e = ilogbf(hu); hb = max(1, min(7, e + hboff)); }
    ctx = NSB_MAX + hb * 8 + mb;
}

struct TM { long long blk0, seed_off; int seed_type, cols; };  // per tensor of the batch
struct SG { int t, nb; long long b0; };                       // segment: blocks [b0, b0+nb) of tensor t (block ids within the batch)

#define PUT(tab, sym) {                                                               \
    int tt_ = remap[tab]; unsigned int s_ = (sym);                                    \
    unsigned int f_ = freq[tt_ * 256 + s_], c_ = cum[tt_ * 257 + s_];                 \
    unsigned int xm_ = ((RANS_L >> PB) << 8) * f_;                                    \
    while (x >= xm_) { *--ptr = (uchar)(x & 0xff); x >>= 8; }                         \
    x = ((x / f_) << PB) + (x % f_) + c_; }

#define GET(tab, out) {                                                              \
    int tt_ = remap[tab]; unsigned int slot_ = x & (PSCALE - 1);                      \
    int s_ = lut[(size_t)tt_ * PSCALE + slot_];                                       \
    x = freq[tt_ * 256 + s_] * (x >> PB) + slot_ - cum[tt_ * 257 + s_];               \
    while (x < RANS_L) x = (x << 8) | *ptr++;                                         \
    out = s_; }

#define SEG_SETUP                                                                     \
    int s = blockIdx.x * blockDim.x + threadIdx.x;                                    \
    if (s >= nseg) return;                                                            \
    const SG g = sg[s]; const TM m = tm[g.t];                                         \
    const int qk = QKof(T), bs = BSof(T), nsb = NSB(T);                               \
    const uchar* sbase = seed + m.seed_off;                                           \
    const int sublen = SUBLEN(T);                                                     \
    SC scache; scache.key = -1; scache.a = scache.c = 0.f;                                const int hboff = m.seed_type == 98 ? HBOFF_COL : 8;

// histogram add with warp aggregation: lanes adding to the same counter make one atomic
__device__ __forceinline__ void hadd(unsigned int* counts, int key) {
    unsigned int mask = __match_any_sync(__activemask(), key);
    if ((threadIdx.x & 31) == __ffs(mask) - 1) atomicAdd(&counts[key], __popc(mask));
}

extern "C" __global__ void qr_hist(int T, const uchar* F, const uchar* P, const uchar* seed,
                                   const TM* tm, const SG* sg, int nseg, unsigned int* counts) {
    SEG_SETUP
    int prev = 1;
    for (long long b = g.b0; b < g.b0 + g.nb; b++) {
        const uchar* fb = F + b * bs; const uchar* pb = P + b * bs;
        int eq = same(fb, pb, bs);
        hadd(counts, (FLAG + prev) * 256 + eq);
        prev = eq;
        if (eq) continue;
        int any = 0;
        for (int k = 0; k < NFIELD(T); k++) {
            int kd = FKIND(T, k), t = get_field(T, fb, k), pr = get_field(T, pb, k), dlt = t - pr;
            if (kd < 2 && (dlt < -127 || dlt > 127)) {
                hadd(counts, FCTX(kd, any) * 256 + 128);
                hadd(counts, FESC_LO * 256 + (t & 255)); hadd(counts, FESC_HI * 256 + (t >> 8));
            } else hadd(counts, FCTX(kd, any) * 256 + (dlt & 255));
            any |= dlt != 0;
        }
        long long gw0 = (b - m.blk0) * qk;
        int tsub = -1; float ta = 0.f, tc = 0.f;
        for (int w = 0; w < qk; w++) {
            if (w / sublen != tsub) { tsub = w / sublen; wscale(T, fb, w, ta, tc); }
            float xv, hv; seed_val(m.seed_type, sbase, gw0 + w, xv, hv, scache, m.cols);
            int r, sg_, ctx; predict_code(T, ta, tc, pb, w, xv, hv, hboff, r, sg_, ctx);
            int sym = (sg_ * (get_code(T, fb, w) - r)) & 255;
            hadd(counts, ctx * 256 + sym);
        }
    }
}

extern "C" __global__ void qr_enc(int T, const uchar* F, const uchar* P, const uchar* seed,
                                  const TM* tm, const SG* sg, int nseg,
                                  const int* remap, const unsigned short* freq, const unsigned int* cum,
                                  uchar* out, size_t cap, unsigned int* out_len) {
    SEG_SETUP
    uchar* end = out + (size_t)(s + 1) * cap; uchar* ptr = end;
    unsigned int x = RANS_L;
    for (long long b = g.b0 + g.nb - 1; b >= g.b0; b--) {
        const uchar* fb = F + b * bs; const uchar* pb = P + b * bs;
        int eq = same(fb, pb, bs);
        int prev = b > g.b0 ? same(fb - bs, pb - bs, bs) : 1;
        if (eq) { PUT(FLAG + prev, 1); continue; }
        long long gw0 = (b - m.blk0) * qk;
        int tsub = -1; float ta = 0.f, tc = 0.f;
        for (int w = qk - 1; w >= 0; w--) {
            if (w / sublen != tsub) { tsub = w / sublen; wscale(T, fb, w, ta, tc); }
            float xv, hv; seed_val(m.seed_type, sbase, gw0 + w, xv, hv, scache, m.cols);
            int r, sg_, ctx; predict_code(T, ta, tc, pb, w, xv, hv, hboff, r, sg_, ctx);
            PUT(ctx, (sg_ * (get_code(T, fb, w) - r)) & 255);
            if (ptr - out < (long long)s * cap + 8) { out_len[s] = 0xffffffffu; return; }
        }
        {
            int nf = NFIELD(T), anyk[MAXF], dl[MAXF], tv[MAXF], any = 0;
            for (int k = 0; k < nf; k++) { tv[k] = get_field(T, fb, k); dl[k] = tv[k] - get_field(T, pb, k); anyk[k] = any; any |= dl[k] != 0; }
            for (int k = nf - 1; k >= 0; k--) {
                int kd = FKIND(T, k);
                if (kd < 2 && (dl[k] < -127 || dl[k] > 127)) {
                    PUT(FESC_HI, tv[k] >> 8); PUT(FESC_LO, tv[k] & 255); PUT(FCTX(kd, anyk[k]), 128);
                } else PUT(FCTX(kd, anyk[k]), dl[k] & 255);
            }
        }
        PUT(FLAG + prev, 0);
    }
    for (int i = 0; i < 4; i++) { *--ptr = (uchar)(x & 0xff); x >>= 8; }
    out_len[s] = (unsigned int)(end - ptr);
}

extern "C" __global__ void qr_dec(int T, const uchar* in, const unsigned long long* in_off, const uchar* P,
                                  const uchar* seed, const TM* tm, const SG* sg, int nseg,
                                  const int* remap, const unsigned short* freq, const unsigned int* cum,
                                  const uchar* lut, uchar* F) {
    SEG_SETUP
    const uchar* ptr = in + in_off[s]; unsigned int x = 0;
    for (int i = 0; i < 4; i++) x = (x << 8) | *ptr++;
    int prev = 1;
    for (long long b = g.b0; b < g.b0 + g.nb; b++) {
        uchar* fb = F + b * bs; const uchar* pb = P + b * bs;
        int eq; GET(FLAG + prev, eq);
        prev = eq;
        if (eq) { for (int i = 0; i < bs; i++) fb[i] = pb[i]; continue; }
        {
            int vals[MAXF], any = 0;
            for (int k = 0; k < NFIELD(T); k++) {
                int kd = FKIND(T, k), sym, pr = get_field(T, pb, k), t;
                GET(FCTX(kd, any), sym);
                if (kd < 2 && sym == 128) { int lo, hi; GET(FESC_LO, lo); GET(FESC_HI, hi); t = lo | (hi << 8); }
                else {
                    t = pr + (int)(signed char)sym;
                    if (kd < 2) t &= 0xffff;
                    else if (T == 14) t = (int)(signed char)(t & 255);
                }
                vals[k] = t; any |= t != pr;
            }
            set_fields(T, fb, vals);
        }
        long long gw0 = (b - m.blk0) * qk;
        int tsub = -1; float ta = 0.f, tc = 0.f;
        for (int w = 0; w < qk; w++) {
            if (w / sublen != tsub) { tsub = w / sublen; wscale(T, fb, w, ta, tc); }
            float xv, hv; seed_val(m.seed_type, sbase, gw0 + w, xv, hv, scache, m.cols);
            int r, sg_, ctx; predict_code(T, ta, tc, pb, w, xv, hv, hboff, r, sg_, ctx);
            int v; GET(ctx, v);
            int q = r + sg_ * (int)(signed char)v;
            if (T == 8) q = (int)(signed char)(q & 255);
            put_code(T, fb, w, q);
        }
    }
}

// per weight of quantised blocks: the interval of original values that rounds to its code
extern "C" __global__ void intervals(int T, const uchar* F, long long nw, float* lo, float* hi) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nw) return;
    int qk = QKof(T);
    const uchar* b = F + (i / qk) * BSof(T);
    int w = (int)(i % qk), q = get_code(T, b, w);
    float a, c; wscale(T, b, w, a, c);
    float l, h;
    if (NONLIN(T)) {
        float v = __fmul_rn(a, KV[q]);
        l = q > 0 ? __fmul_rn(a, 0.5f * (KV[q - 1] + KV[q])) : -INFINITY;
        h = q < 15 ? __fmul_rn(a, 0.5f * (KV[q] + KV[q + 1])) : INFINITY;
        if (a < 0.f) { float t = l; l = q < 15 ? h : -INFINITY; h = q > 0 ? t : INFINITY; }
        if (a == 0.f) { l = -INFINITY; h = INFINITY; }
        (void)v;
    } else {
        float v = __fadd_rn(__fmul_rn(a, (float)q), c), e = __fmul_rn(fabsf(a), 0.5f);
        l = __fsub_rn(v, e); h = __fadd_rn(v, e);
        bool lowend = q == QMIN(T), highend = q == QMAX(T);
        if (a > 0.f) { if (lowend) l = -INFINITY; if (highend) h = INFINITY; }
        else if (a < 0.f) { if (highend) l = -INFINITY; if (lowend) h = INFINITY; }
        else { l = -INFINITY; h = INFINITY; }
    }
    lo[i] = l; hi[i] = h;
}

// per weight: 1 if its code and its sub-block's integer scale (and min) equal the other block's
extern "C" __global__ void weq(int T, const uchar* F, const uchar* P, long long nw, uchar* out) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nw) return;
    int qk = QKof(T);
    const uchar* f = F + (i / qk) * BSof(T); const uchar* p = P + (i / qk) * BSof(T);
    int w = (int)(i % qk);
    int eq = get_code(T, f, w) == get_code(T, p, w);
    int sub = w / SUBLEN(T);
    switch (T) {
    case 10: eq &= f[sub] == p[sub]; break;
    case 11: eq &= q3k_scale(f + 96, sub) == q3k_scale(p + 96, sub); break;
    case 12: case 13: { int s1, m1, s2, m2; scale_min_k4(sub, f + 4, s1, m1); scale_min_k4(sub, p + 4, s2, m2); eq &= s1 == s2 && m1 == m2; break; }
    case 14: eq &= f[192 + sub] == p[192 + sub]; break;
    case 23: eq &= get_field(T, f, 1 + sub) == get_field(T, p, 1 + sub); break;
    }
    out[i] = (uchar)eq;
}

// scale fields of every block (no-seed coding statistics)
extern "C" __global__ void block_fields(int T, const uchar* F, long long nblk, int* out) {
    long long b = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblk) return;
    int nf = NFIELD(T);
    for (int f = 0; f < nf; f++) out[b * nf + f] = get_field(T, F + b * BSof(T), f);
}
// blocks with the given field values and zero codes
extern "C" __global__ void blocks_from_fields(int T, const int* v, int n, uchar* out) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    uchar* o = out + (long long)k * BSof(T);
    for (int i = 0; i < BSof(T); i++) o[i] = 0;
    set_fields(T, o, v + (long long)k * NFIELD(T));
}

extern "C" __global__ void gather(const uchar* src, size_t cap, const unsigned int* len,
                                  const unsigned long long* off, int nseg, uchar* dst) {
    int s = blockIdx.x;
    if (s >= nseg) return;
    const uchar* a = src + (size_t)(s + 1) * cap - len[s];
    uchar* d = dst + off[s];
    for (unsigned int i = threadIdx.x; i < len[s]; i += blockDim.x) d[i] = a[i];
}

extern "C" __global__ void build_lut(const unsigned int* cum, int ntab, uchar* lut) {
    int tb = blockIdx.x;
    if (tb >= ntab) return;
    for (int sym = threadIdx.x; sym < 256; sym += blockDim.x)
        for (unsigned int i = cum[tb * 257 + sym]; i < cum[tb * 257 + sym + 1]; i++)
            lut[(size_t)tb * PSCALE + i] = (uchar)sym;
}
"""

TM_DT = np.dtype([("blk0", "<i8"), ("seed_off", "<i8"), ("seed_type", "<i4"), ("cols", "<i4")])
COLRMS = 98                              # no seed: x = 0, uncertainty = the column's RMS (float32 per column)
SG_DT = np.dtype([("t", "<i4"), ("nb", "<i4"), ("b0", "<i8")])
_mod = None


def normalize(counts, scale=1 << PB):
    """counts (T, 256) -> (uint16 freqs of the used rows, used row ids), each row summing to `scale`
    and every seen symbol >= 1. On the host: the contexts here can hold one dominant symbol and
    hundreds of rare ones, where gpu2.normalize_gpu's iteration does not converge."""
    c = cp.asnumpy(counts).astype(np.int64)
    tot = c.sum(1)
    used = np.flatnonzero(tot)
    f = np.zeros((len(used), 256), np.int64)
    for k, r in enumerate(used):
        row = c[r]
        fr = np.where(row > 0, np.maximum(1, row * scale // tot[r]), 0)
        diff = scale - fr.sum()
        while diff != 0:
            j = int(np.argmax(fr))
            step = diff if diff > 0 else -min(-diff, fr[j] - 1)
            if step == 0:                         # largest is 1: take from any entry > 1
                j = int(np.flatnonzero(fr > 1)[0]); step = -1
            fr[j] += step; diff -= step
        f[k] = fr
    return cp.asarray(f.astype(np.uint16)), cp.asarray(used.astype(np.int32))


def kern(name):
    global _mod
    if _mod is None:
        _mod = cp.RawModule(code=KERNELS, options=("-std=c++17", "--fmad=false"))
    return _mod.get_function(name)


def seg_blocks(T):
    """blocks per rANS segment (= per GPU thread): ~16K symbols"""
    return max(16, 16384 // QK[T])


def _layout(T, nblocks):
    """tensors' block counts -> (TM rows without seed fields, SG array, total blocks)"""
    sb = seg_blocks(T)
    tm = np.zeros(len(nblocks), TM_DT)
    segs, b0 = [], 0
    for i, nb in enumerate(nblocks):
        tm[i]["blk0"] = b0
        for o in range(0, nb, sb):
            segs.append((i, min(sb, nb - o), b0 + o))
        b0 += nb
    sg = np.array(segs, SG_DT) if segs else np.zeros(0, SG_DT)
    return tm, sg, b0


def batches(T, sizes):
    """split tensor indices into coding batches of about BATCH bytes (same split on both sides)"""
    out, cur, n = [], [], 0
    for i, s in enumerate(sizes):
        if cur and n + s > BATCH:
            out.append(cur); cur, n = [], 0
        cur.append(i); n += s
    if cur:
        out.append(cur)
    return out


def encode(T, items, seed_dev):
    """items: [(file tensor bytes, predicted bytes, seed byte offset, seed type)] of one batch
    (all of type T; host or device uint8 arrays) -> bytes"""
    bs = BS[T]
    nbs = [len(it[0]) // bs for it in items]
    tm, sg, nb = _layout(T, nbs)
    tm["seed_off"] = [it[2] for it in items]
    tm["seed_type"] = [it[3] for it in items]
    tm["cols"] = [it[4] if len(it) > 4 else 0 for it in items]
    F = cp.concatenate([cp.asarray(it[0]) for it in items])
    P = cp.concatenate([cp.asarray(it[1]) for it in items])
    tm_g, sg_g = cp.asarray(tm.view(np.uint8)), cp.asarray(sg.view(np.uint8))
    nseg = len(sg)
    counts = cp.zeros((NTAB, 256), cp.uint32)
    grid = ((nseg + 63) // 64,)
    kern("qr_hist")(grid, (64,), (np.int32(T), F, P, seed_dev, tm_g, sg_g, np.int32(nseg), counts))
    if STATS is not None:
        c = cp.asnumpy(counts).astype(np.float64)
        for name, rows in (("flags", range(NSB_MAX + 64, NTAB)), ("scales", range(NSB_MAX)), ("codes", range(NSB_MAX, NSB_MAX + 64))):
            sub = c[list(rows)]
            tot = sub.sum(1, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                bits = -np.nansum(np.where(sub > 0, sub * np.log2(sub / tot), 0))
            STATS[(T, name)] = STATS.get((T, name), 0) + bits / 8
    freq, used = normalize(counts)
    remap = cp.full(NTAB, -1, cp.int32); remap[used] = cp.arange(len(used), dtype=cp.int32)
    cum = cum_of(freq)
    cap = seg_blocks(T) * (QK[T] + NSB_MAX) * 2 + 64
    out = cp.empty(nseg * cap, cp.uint8); out_len = cp.zeros(nseg, cp.uint32)
    kern("qr_enc")(grid, (64,), (np.int32(T), F, P, seed_dev, tm_g, sg_g, np.int32(nseg), remap, freq, cum,
                                 out, np.uint64(cap), out_len))
    lens = out_len.get()
    if (lens == 0xffffffff).any():
        raise RuntimeError("QR segment overflow")
    offs = cp.zeros(nseg + 1, cp.uint64); offs[1:] = cp.cumsum(out_len.astype(cp.uint64))
    dst = cp.empty(int(offs[-1]), cp.uint8)
    kern("gather")((max(nseg, 1),), (256,), (out, np.uint64(cap), out_len, offs, np.int32(nseg), dst))
    stream = dst.get().tobytes()
    return pack({"t": store_tables(freq, used), "l": zstd_c(lens.astype(np.uint32).tobytes(), 19), "s": stream})


def decode(T, buf, items, seed_dev):
    """items: [(predicted bytes, seed byte offset, seed type)] -> device uint8 array of the batch's
    tensors back to back (same order)"""
    ch = unpack(buf)
    bs = BS[T]
    nbs = [len(it[0]) // bs for it in items]
    tm, sg, nb = _layout(T, nbs)
    tm["seed_off"] = [it[1] for it in items]
    tm["seed_type"] = [it[2] for it in items]
    tm["cols"] = [it[3] if len(it) > 3 else 0 for it in items]
    P = cp.concatenate([cp.asarray(it[0]) for it in items])
    F = cp.zeros(nb * bs, cp.uint8)
    nseg = len(sg)
    freq_h, used_h = load_tables(ch["t"])
    freq = cp.asarray(freq_h); cum = cum_of(freq)
    lut = cp.empty((len(freq_h), 1 << PB), cp.uint8)
    kern("build_lut")((len(freq_h),), (256,), (cum, np.int32(len(freq_h)), lut))
    remap = cp.full(NTAB, -1, cp.int32); remap[cp.asarray(used_h)] = cp.arange(len(used_h), dtype=cp.int32)
    lens = np.frombuffer(zstd_d(ch["l"]), np.uint32).astype(np.uint64)
    offs = np.zeros(nseg, np.uint64); offs[1:] = np.cumsum(lens)[:-1]
    kern("qr_dec")(((nseg + 63) // 64,), (64,),
                   (np.int32(T), cp.asarray(np.frombuffer(ch["s"], np.uint8)), cp.asarray(offs), P, seed_dev,
                    cp.asarray(tm.view(np.uint8)), cp.asarray(sg.view(np.uint8)), np.int32(nseg),
                    remap, freq, cum, lut, F))
    return F


def intervals(T, blocks, nw):
    """device uint8 blocks of type T -> (lo, hi) float32 device arrays of the value intervals"""
    lo = cp.empty(nw, cp.float32); hi = cp.empty(nw, cp.float32)
    kern("intervals")(((nw + 255) // 256,), (256,), (np.int32(T), blocks, np.int64(nw), lo, hi))
    return lo, hi


def weight_equal(T, F, P, nw):
    """device uint8 per weight: code and sub-block integer scale equal in blocks F and P"""
    out = cp.empty(nw, cp.uint8)
    kern("weq")(((nw + 255) // 256,), (256,), (np.int32(T), F, P, np.int64(nw), out))
    return out


def self_side(T, blocks, cols):
    """no-seed coding of one tensor (device uint8 blocks, `cols` weights per row) -> side info
    (column RMS as fp16 bytes, template blocks) and the device arrays the coder uses"""
    from .gpuquant import GPUQuant
    gq = _gq()
    x = gq.dequantize(T, blocks, cols).reshape(-1, cols)
    rms = cp.sqrt((x * x).mean(0)).astype(cp.float16)
    del x
    nb_row = cols // QK[T]
    rows = len(blocks) // BS[T] // nb_row
    nf = NFIELDS[T]
    nblk = rows * nb_row
    fl = cp.empty(nblk * nf, cp.int32)
    kern("block_fields")(((nblk + 255) // 256,), (256,), (np.int32(T), blocks, np.int64(nblk), fl))
    v = cp.rint(fl.reshape(rows, nb_row, nf).astype(cp.float64).mean(0)).astype(cp.int32)
    tpl = cp.empty(nb_row * BS[T], cp.uint8)
    kern("blocks_from_fields")(((nb_row + 31) // 32,), (32,), (np.int32(T), v, np.int32(nb_row), tpl))
    return rms, tpl, rows


_GQ = None


def _gq():
    global _GQ
    if _GQ is None:
        from .gpuquant import GPUQuant
        _GQ = GPUQuant()
    return _GQ


def self_encode(T, tensors):
    """tensors: [(host or device uint8 blocks, cols)] of type T, coded without a seed -> bytes"""
    parts, side = {}, []
    for bi in batches(T, [len(b) for b, _ in tensors]):
        items, seeds, rms_l, tpl_l, o = [], [], [], [], 0
        for i in bi:
            b, cols = tensors[i]
            bd = cp.asarray(b)
            rms, tpl, rows = self_side(T, bd, cols)
            rms_l.append(rms.get()); tpl_l.append(tpl.get())
            seeds.append(rms.astype(cp.float32))
            items.append((bd, cp.tile(tpl, rows), o, COLRMS, cols)); o += cols * 4
        parts[str(len(side))] = encode(T, items, cp.concatenate(seeds).view(cp.uint8))
        side.append([[len(tensors[i][0]), tensors[i][1]] for i in bi])
        parts[f"r{len(side) - 1}"] = zstd_c(b"".join(r.tobytes() for r in rms_l), 19)
        parts[f"t{len(side) - 1}"] = zstd_c(b"".join(t.tobytes() for t in tpl_l), 19)
    return pack({"meta": zstd_c(json.dumps(side).encode()), **parts})


def self_decode(T, buf):
    """-> list of device uint8 arrays, in self_encode's order"""
    ch = unpack(buf)
    out = []
    for k, sizes in enumerate(json.loads(zstd_d(ch["meta"]))):
        rms_all = np.frombuffer(zstd_d(ch[f"r{k}"]), np.float16)
        tpl_all = np.frombuffer(zstd_d(ch[f"t{k}"]), np.uint8)
        items, seeds, ro, to, o = [], [], 0, 0, 0
        for nbytes, cols in sizes:
            nb_row = cols // QK[T]; rows = nbytes // BS[T] // nb_row
            rms = cp.asarray(rms_all[ro:ro + cols]); ro += cols
            tpl = cp.asarray(tpl_all[to:to + nb_row * BS[T]]); to += nb_row * BS[T]
            seeds.append(rms.astype(cp.float32))
            items.append((cp.tile(tpl, rows), o, COLRMS, cols)); o += cols * 4
        F = decode(T, ch[str(k)], items, cp.concatenate(seeds).view(cp.uint8))
        p = 0
        for nbytes, _ in sizes:
            out.append(F[p:p + nbytes]); p += nbytes
    return out
