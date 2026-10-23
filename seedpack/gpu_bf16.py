"""Methods 24-25: GPU ports of M22 (BF16 codec) and M23 (model-repo bundle).

M24: same model as M22 (exponent | predicted exponent, mantissa | exponent offset,
     sign | column bias class). Weights are cut into segments of SEG elements inside each
     tensor; one CUDA thread codes one segment with its own rANS state (interleaved
     exponent / mantissa / sign symbols). Contexts are computed in the kernels from the
     integer side info, so encoder and decoder agree bit for bit.
M25: bundle = BF16 base (M24) + Q8_0 files coded against the base that the decoder has
     just reconstructed on the GPU. Per Q8_0 block: scale delta vs fp16(max|w|/127) (with
     escape), weights as (q - rint(w/d)) mod 256 with context = reference precision x |pred|.
     fp16 rounding and floor(2*log2) are done with exact bit arithmetic.
"""
import json
import struct
from concurrent.futures import ThreadPoolExecutor

import cupy as cp
import numpy as np
import zstandard

from .codecs_gguf import pack, unpack, zstd_c, zstd_d
from .gguf_reader import GGUF, GGML_BF16, GGML_Q8_0
from .gpu2 import (header_pairs_c, header_pairs_d, xz_chunks_c, xz_chunks_d, rest_of,
                  normalize_gpu, cum_of, store_tables, load_tables, PB, Timer, GPU2)

SEG = 16384            # bf16 weights per thread
SEGQ = 512             # Q8_0 blocks per thread (M25)
TPT = 160              # tables per bf16 tensor: 128 exponent + 16 mantissa + 16 sign
TPQ = 35               # tables per Q8_0 tensor: 32 residual + 1 scale delta + 2 escape bytes

KERNELS = r"""
#define PB 12
#define PSCALE (1u << PB)
#define RANS_L (1u << 23)
#define SEG %(SEG)d
#define SEGQ %(SEGQ)d

struct TB { long long off; int rowlen, rowbase, colbase, tab; };     // bf16 tensor
struct SG { int t; int n; long long i0; };                            // segment

#define PUT(tab, s) {                                                                 \
    int tt_ = remap[tab]; unsigned int s_ = (s);                                       \
    unsigned int f_ = freq[tt_ * 256 + s_], c_ = cum[tt_ * 257 + s_];                  \
    unsigned int xm_ = ((RANS_L >> PB) << 8) * f_;                                     \
    while (x >= xm_) { *--ptr = (unsigned char)(x & 0xff); x >>= 8; }                  \
    x = ((x / f_) << PB) + (x %% f_) + c_; }

#define GET(tab, out) {                                                               \
    int tt_ = remap[tab]; unsigned int slot_ = x & (PSCALE - 1);                       \
    int s_ = lut[(size_t)tt_ * PSCALE + slot_];                                        \
    x = freq[tt_ * 256 + s_] * (x >> PB) + slot_ - cum[tt_ * 257 + s_];                \
    while (x < RANS_L) x = (x << 8) | *ptr++;                                          \
    out = s_; }

__device__ __forceinline__ int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

// ---------------------------------------------------------------- M24: bf16
#define BF_CTX                                                                         \
    int pred4 = rfi[T.rowbase + r] + cfi[T.colbase + j];                               \
    int eb = clampi((pred4 + 108) >> 1, 0, 127);

extern "C" __global__ void bf_hist(const unsigned char* file, const TB* tb, const SG* sg, int nseg,
                                   const short* rfi, const short* cfi, const signed char* scl,
                                   unsigned int* counts) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nseg) return;
    TB T = tb[sg[s].t];
    const unsigned short* w = (const unsigned short*)(file + T.off) + sg[s].i0;
    int r = (int)(sg[s].i0 / T.rowlen), j = (int)(sg[s].i0 %% T.rowlen);
    for (int k = 0; k < sg[s].n; k++) {
        unsigned int u = w[k];
        int ex = (u >> 7) & 255, man = u & 127, sgn = u >> 15;
        BF_CTX
        int rel = clampi(ex - ((pred4 + 508) >> 2) + 8, 0, 15);
        atomicAdd(&counts[(size_t)(T.tab + eb) * 256 + ex], 1u);
        atomicAdd(&counts[(size_t)(T.tab + 128 + rel) * 256 + man], 1u);
        atomicAdd(&counts[(size_t)(T.tab + 144 + scl[T.colbase + j]) * 256 + sgn], 1u);
        if (++j == T.rowlen) { j = 0; r++; }
    }
}

extern "C" __global__ void bf_enc(const unsigned char* file, const TB* tb, const SG* sg, int nseg,
                                  const short* rfi, const short* cfi, const signed char* scl,
                                  const int* remap, const unsigned short* freq, const unsigned int* cum,
                                  unsigned char* out, size_t cap, unsigned int* out_len) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nseg) return;
    TB T = tb[sg[s].t];
    const unsigned short* w = (const unsigned short*)(file + T.off) + sg[s].i0;
    long long last = sg[s].i0 + sg[s].n - 1;
    int r = (int)(last / T.rowlen), j = (int)(last %% T.rowlen);
    unsigned char* start = out + (size_t)s * cap;
    unsigned char* end = start + cap; unsigned char* ptr = end;
    unsigned int x = RANS_L;
    for (int k = sg[s].n - 1; k >= 0; k--) {
        unsigned int u = w[k];
        int ex = (u >> 7) & 255, man = u & 127, sgn = u >> 15;
        BF_CTX
        int rel = clampi(ex - ((pred4 + 508) >> 2) + 8, 0, 15);
        PUT(T.tab + 144 + scl[T.colbase + j], sgn);
        PUT(T.tab + 128 + rel, man);
        PUT(T.tab + eb, ex);
        if (ptr - start < 64) { out_len[s] = 0xffffffffu; return; }      // slot overflow
        if (--j < 0) { j = T.rowlen - 1; r--; }
    }
    for (int i = 0; i < 4; i++) { *--ptr = (unsigned char)(x & 0xff); x >>= 8; }
    out_len[s] = (unsigned int)(end - ptr);
}

extern "C" __global__ void bf_dec(const unsigned char* in, const unsigned long long* in_off,
                                  unsigned char* file, const TB* tb, const SG* sg, int nseg,
                                  const short* rfi, const short* cfi, const signed char* scl,
                                  const int* remap, const unsigned short* freq, const unsigned int* cum,
                                  const unsigned char* lut) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nseg) return;
    TB T = tb[sg[s].t];
    unsigned short* w = (unsigned short*)(file + T.off) + sg[s].i0;
    int r = (int)(sg[s].i0 / T.rowlen), j = (int)(sg[s].i0 %% T.rowlen);
    const unsigned char* ptr = in + in_off[s]; unsigned int x = 0;
    for (int i = 0; i < 4; i++) x = (x << 8) | *ptr++;
    for (int k = 0; k < sg[s].n; k++) {
        int ex, man, sgn;
        BF_CTX
        GET(T.tab + eb, ex);
        int rel = clampi(ex - ((pred4 + 508) >> 2) + 8, 0, 15);
        GET(T.tab + 128 + rel, man);
        GET(T.tab + 144 + scl[T.colbase + j], sgn);
        w[k] = (unsigned short)((sgn << 15) | (ex << 7) | man);
        if (++j == T.rowlen) { j = 0; r++; }
    }
}

// ---------------------------------------------------------------- M25: Q8_0 against a bf16 base
struct TQ { long long qoff, boff; int nblocks, tab; };
struct SQ { int t, b0, n, pad; };

__device__ __forceinline__ float bf2f(unsigned short u) { return __uint_as_float(((unsigned int)u) << 16); }

__device__ __forceinline__ float h2f(unsigned int h) {
    unsigned int s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023, f;
    if (e == 0) {
        if (m == 0) f = s << 31;
        else { int ee = 113; while (!(m & 1024)) { m <<= 1; ee--; } m &= 1023; f = (s << 31) | (ee << 23) | (m << 13); }
    } else if (e == 31) f = (s << 31) | 0x7f800000u | (m << 13);
    else f = (s << 31) | ((e + 112) << 23) | (m << 13);
    return __int_as_float(f);
}

// float -> fp16 bits, round to nearest even (exact bit arithmetic)
__device__ __forceinline__ unsigned int f2h(float x) {
    unsigned int f = __float_as_uint(x), sign = (f >> 16) & 0x8000u, m = f & 0x7fffffu;
    int e = (int)((f >> 23) & 255);
    if (e == 255) return sign | 0x7c00u | (m ? 0x200u : 0u);
    e = e - 127 + 15;
    if (e >= 31) return sign | 0x7c00u;
    if (e <= 0) {
        if (e < -10) return sign;
        m |= 0x800000u;
        int shift = 14 - e;
        unsigned int hm = m >> shift, rem = m & ((1u << shift) - 1), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (hm & 1))) hm++;
        return sign | hm;
    }
    unsigned int hm = m >> 13, rem = m & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (hm & 1))) {
        hm++;
        if (hm == 0x400u) { hm = 0; e++; if (e >= 31) return sign | 0x7c00u; }
    }
    return sign | ((unsigned int)e << 10) | hm;
}

// floor(2 * log2(x)) for x > 0, exact
__device__ __forceinline__ int floor2log2(float x) {
    unsigned int f = __float_as_uint(x);
    int e = (int)((f >> 23) & 255) - 127;
    return 2 * e + ((f & 0x7fffffu) >= 0x3504f3u ? 1 : 0);    // 0x3504f3 = mantissa of sqrt(2)
}

#define Q_BLOCK_SETUP                                                                  \
    const unsigned short* wr = (const unsigned short*)(base + T.boff) + (size_t)b * 32; \
    float amax = 0.0f;                                                                 \
    for (int k = 0; k < 32; k++) amax = fmaxf(amax, fabsf(bf2f(wr[k])));               \
    int dpred = (int)f2h(__fdiv_rn(amax, 127.0f));

#define Q_WEIGHT_CTX(k)                                                                \
    float wv = __fmul_rn(bf2f(wr[k]), inv);                                            \
    int pred = (int)fmaxf(-127.0f, fminf(127.0f, rintf(wv)));                          \
    float ax = fabsf(wv);                                                              \
    int ebin = ax > 0.0f ? clampi(floor2log2(ax) - 14 + 6, 0, 7) : 0;                  \
    int ctx = ebin * 4 + min(abs(pred) / 32, 3);

extern "C" __global__ void q_hist(const unsigned char* qfile, const unsigned char* base,
                                  const TQ* tq, const SQ* sq, int nseg, unsigned int* counts) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nseg) return;
    TQ T = tq[sq[s].t];
    for (int b = sq[s].b0; b < sq[s].b0 + sq[s].n; b++) {
        const unsigned char* blk = qfile + T.qoff + (size_t)b * 34;
        Q_BLOCK_SETUP
        int dbits = blk[0] | (blk[1] << 8), delta = dbits - dpred;
        if (delta > -127 && delta < 127) atomicAdd(&counts[(size_t)(T.tab + 32) * 256 + delta + 128], 1u);
        else {
            atomicAdd(&counts[(size_t)(T.tab + 32) * 256 + 0], 1u);
            atomicAdd(&counts[(size_t)(T.tab + 33) * 256 + blk[1]], 1u);
            atomicAdd(&counts[(size_t)(T.tab + 34) * 256 + blk[0]], 1u);
        }
        float d = h2f(dbits), inv = d > 0.0f ? __fdiv_rn(1.0f, d) : 0.0f;
        for (int k = 0; k < 32; k++) {
            Q_WEIGHT_CTX(k)
            int sym = ((signed char)blk[2 + k] - pred) & 255;
            atomicAdd(&counts[(size_t)(T.tab + ctx) * 256 + sym], 1u);
        }
    }
}

extern "C" __global__ void q_enc(const unsigned char* qfile, const unsigned char* base,
                                 const TQ* tq, const SQ* sq, int nseg,
                                 const int* remap, const unsigned short* freq, const unsigned int* cum,
                                 unsigned char* out, size_t cap, unsigned int* out_len) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nseg) return;
    TQ T = tq[sq[s].t];
    unsigned char* start = out + (size_t)s * cap;
    unsigned char* end = start + cap; unsigned char* ptr = end;
    unsigned int x = RANS_L;
    for (int b = sq[s].b0 + sq[s].n - 1; b >= sq[s].b0; b--) {
        const unsigned char* blk = qfile + T.qoff + (size_t)b * 34;
        Q_BLOCK_SETUP
        int dbits = blk[0] | (blk[1] << 8), delta = dbits - dpred;
        float d = h2f(dbits), inv = d > 0.0f ? __fdiv_rn(1.0f, d) : 0.0f;
        for (int k = 31; k >= 0; k--) {
            Q_WEIGHT_CTX(k)
            PUT(T.tab + ctx, ((signed char)blk[2 + k] - pred) & 255);
        }
        if (delta > -127 && delta < 127) PUT(T.tab + 32, delta + 128)
        else { PUT(T.tab + 34, blk[0]); PUT(T.tab + 33, blk[1]); PUT(T.tab + 32, 0); }
        if (ptr - start < 128) { out_len[s] = 0xffffffffu; return; }
    }
    for (int i = 0; i < 4; i++) { *--ptr = (unsigned char)(x & 0xff); x >>= 8; }
    out_len[s] = (unsigned int)(end - ptr);
}

extern "C" __global__ void q_dec(const unsigned char* in, const unsigned long long* in_off,
                                 unsigned char* qfile, const unsigned char* base,
                                 const TQ* tq, const SQ* sq, int nseg,
                                 const int* remap, const unsigned short* freq, const unsigned int* cum,
                                 const unsigned char* lut) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nseg) return;
    TQ T = tq[sq[s].t];
    const unsigned char* ptr = in + in_off[s]; unsigned int x = 0;
    for (int i = 0; i < 4; i++) x = (x << 8) | *ptr++;
    for (int b = sq[s].b0; b < sq[s].b0 + sq[s].n; b++) {
        unsigned char* blk = qfile + T.qoff + (size_t)b * 34;
        Q_BLOCK_SETUP
        int ds, dbits;
        GET(T.tab + 32, ds);
        if (ds == 0) { int hi, lo; GET(T.tab + 33, hi); GET(T.tab + 34, lo); dbits = lo | (hi << 8); }
        else dbits = (dpred + ds - 128) & 0xffff;
        blk[0] = dbits & 255; blk[1] = dbits >> 8;
        float d = h2f(dbits), inv = d > 0.0f ? __fdiv_rn(1.0f, d) : 0.0f;
        for (int k = 0; k < 32; k++) {
            Q_WEIGHT_CTX(k)
            int sym; GET(T.tab + ctx, sym);
            blk[2 + k] = (unsigned char)((pred + sym) & 255);
        }
    }
}
""" % dict(SEG=SEG, SEGQ=SEGQ)

_mod = None


def kern(name):
    global _mod
    if _mod is None:
        _mod = cp.RawModule(code=KERNELS, options=("-std=c++17", "--fmad=false"))
    return _mod.get_function(name)


TB_DT = np.dtype([("off", "<i8"), ("rowlen", "<i4"), ("rowbase", "<i4"), ("colbase", "<i4"), ("tab", "<i4")])
SG_DT = np.dtype([("t", "<i4"), ("n", "<i4"), ("i0", "<i8")])
TQ_DT = np.dtype([("qoff", "<i8"), ("boff", "<i8"), ("nblocks", "<i4"), ("tab", "<i4")])
SQ_DT = np.dtype([("t", "<i4"), ("b0", "<i4"), ("n", "<i4"), ("pad", "<i4")])

_gather = GPU2()


def _rans_encode(kname, grid_n, args, counts, cap):
    """Shared driver: tables from counts, encode, gather. Returns (stream bytes, lens, tables chunk)."""
    freq, used = normalize_gpu(counts)
    remap = cp.full(counts.shape[0], -1, cp.int32); remap[used] = cp.arange(len(used), dtype=cp.int32)
    cum = cum_of(freq)
    out = cp.empty(grid_n * cap, cp.uint8); out_len = cp.zeros(grid_n, cp.uint32)
    kern(kname)(((grid_n + 63) // 64,), (64,), args + (remap, freq, cum, out, np.uint64(cap), out_len))
    lens = out_len.get()
    if (lens == 0xffffffff).any():
        raise RuntimeError("rANS segment overflow")
    offs = cp.zeros(grid_n + 1, cp.uint64); offs[1:] = cp.cumsum(out_len.astype(cp.uint64))
    dst = cp.empty(int(offs[-1]), cp.uint8)
    _gather.kern("gather")((grid_n,), (256,), (out, np.uint64(cap), out_len, offs, np.int32(grid_n), dst))
    del out
    return dst.get().tobytes(), lens, store_tables(freq, used)


def _rans_tables(tables_chunk, ntab):
    freq_h, used_h = load_tables(tables_chunk)
    freq = cp.asarray(freq_h); cum = cum_of(freq)
    lut = cp.empty((len(freq_h), 1 << PB), cp.uint8)
    _gather.kern("build_lut")((len(freq_h),), (256,), (cum, np.int32(len(freq_h)), lut))
    remap = cp.full(ntab, -1, cp.int32); remap[cp.asarray(used_h)] = cp.arange(len(used_h), dtype=cp.int32)
    return remap, freq, cum, lut


def _offsets(lens):
    offs = np.zeros(len(lens), np.uint64); offs[1:] = np.cumsum(lens.astype(np.uint64))[:-1]
    return cp.asarray(offs)


# ====================================================================== M24
class M24BF16GPU:
    name = "M24 BF16 context rANS on GPU (port of M22)"
    timing = None

    def __init__(self):
        self.timing = {}

    def _layout(self, g, size):
        ts = sorted((t for t in g.tensors if t.type == GGML_BF16 and len(t.shape) == 2), key=lambda t: t.offset)
        return {"size": size, "t": [[t.offset, t.nbytes, t.shape[0], "bf16"] for t in ts]}

    @staticmethod
    def _tables(layout):
        tb = np.zeros(len(layout["t"]), TB_DT); segs = []
        rowbase = colbase = 0
        for i, (off, nbytes, rowlen, _) in enumerate(layout["t"]):
            n = nbytes // 2
            tb[i] = (off, rowlen, rowbase, colbase, i * TPT)
            for i0 in range(0, n, SEG):
                segs.append((i, min(SEG, n - i0), i0))
            rowbase += n // rowlen; colbase += rowlen
        return tb, np.array(segs, SG_DT)

    def compress_dev(self, data, dev=None):
        """-> chunks dict; `dev` = the file already on the GPU (optional)."""
        ev = Timer()
        g = GGUF.from_bytes(data)
        layout = self._layout(g, len(data))
        hdr_job = ThreadPoolExecutor(1).submit(self._header_c, data, g, layout)
        if dev is None:
            dev = cp.asarray(np.frombuffer(data, np.uint8))
        ev("upload")
        rfi_l, cfi_l, sc_l = [], [], []
        for off, nbytes, rowlen, _ in layout["t"]:
            u = dev[off:off + nbytes].view(cp.uint16).reshape(-1, rowlen)
            w = (u.astype(cp.uint32) << 16).view(cp.float32)
            rf = cp.log2(cp.sqrt((w * w).mean(1, dtype=cp.float64)) + 1e-30)
            wn = w / cp.exp2(rf).astype(cp.float32)[:, None]
            cf = cp.log2(cp.sqrt((wn * wn).mean(0, dtype=cp.float64)) + 1e-30)
            rfi = cp.clip(cp.rint(rf * 4), -32768, 32767).astype(cp.int16)
            cfi = cp.clip(cp.rint(cf * 4), -32768, 32767).astype(cp.int16)
            bias = wn.mean(0, dtype=cp.float64) / cp.exp2(cfi.astype(cp.float64) / 4)
            rfi_l.append(rfi); cfi_l.append(cfi)
            sc_l.append(cp.clip(cp.floor(bias * 16) + 8, 0, 15).astype(cp.int8))
            del u, w, wn
        rfi, cfi, scl = cp.concatenate(rfi_l), cp.concatenate(cfi_l), cp.concatenate(sc_l)
        tb, sg = self._tables(layout)
        tb_g, sg_g = cp.asarray(tb.view(np.uint8)), cp.asarray(sg.view(np.uint8))
        ev("statistics")
        args = (dev, tb_g, sg_g, np.int32(len(sg)), rfi, cfi, scl)
        counts = cp.zeros((len(tb) * TPT, 256), cp.uint32)
        kern("bf_hist")(((len(sg) + 63) // 64,), (64,), args + (counts,))
        stream, lens, tables = _rans_encode("bf_enc", len(sg), args, counts, SEG * 2 + 1024)
        ev("histogram + encode")
        ch = {
            "layout": zstd_c(json.dumps(layout).encode()),
            "side": zstd_c(rfi.get().tobytes() + cfi.get().tobytes() + scl.get().tobytes(), 19),
            "tables": tables,
            "lens": zstd_c(lens.astype(np.uint32).tobytes()),
            "stream": stream,
            "rest": hdr_job.result(),
        }
        ev("header")
        self.timing["compress"] = ev.parts
        return ch

    def decompress_dev(self, ch):
        """-> the whole file as a device array."""
        ev = Timer()
        layout = json.loads(zstd_d(ch["layout"]))
        hdr_job = ThreadPoolExecutor(1).submit(self._header_d, ch["rest"])
        tb, sg = self._tables(layout)
        nrow = int(sum(nb // 2 // rl for _, nb, rl, _ in layout["t"])); ncol = int(sum(rl for _, _, rl, _ in layout["t"]))
        side = zstd_d(ch["side"])
        rfi = cp.asarray(np.frombuffer(side, np.int16, nrow, 0))
        cfi = cp.asarray(np.frombuffer(side, np.int16, ncol, 2 * nrow))
        scl = cp.asarray(np.frombuffer(side, np.int8, ncol, 2 * nrow + 2 * ncol))
        remap, freq, cum, lut = _rans_tables(ch["tables"], len(tb) * TPT)
        lens = np.frombuffer(zstd_d(ch["lens"]), np.uint32)
        stream = cp.asarray(np.frombuffer(ch["stream"], np.uint8))
        dev = cp.empty(layout["size"], cp.uint8)
        ev("unpack + upload")
        kern("bf_dec")(((len(sg) + 63) // 64,), (64,),
                       (stream, _offsets(lens), dev, cp.asarray(tb.view(np.uint8)), cp.asarray(sg.view(np.uint8)),
                        np.int32(len(sg)), rfi, cfi, scl, remap, freq, cum, lut))
        ev("decode")
        rest = cp.asarray(np.frombuffer(hdr_job.result(), np.uint8))
        pos = rpos = 0
        for off, nbytes, _, _ in layout["t"]:
            gap = off - pos
            dev[pos:off] = rest[rpos:rpos + gap]; rpos += gap; pos = off + nbytes
        dev[pos:] = rest[rpos:]
        ev("header")
        self.timing["decompress"] = ev.parts
        return dev

    @staticmethod
    def _header_c(data, g, layout):
        h = header_pairs_c(data, g, layout)
        return b"P" + h if h is not None else b"X" + xz_chunks_c(rest_of(data, layout))

    @staticmethod
    def _header_d(buf):
        return header_pairs_d(buf[1:]) if buf[:1] == b"P" else xz_chunks_d(buf[1:])

    def compress(self, data):
        return pack(self.compress_dev(data))

    def decompress(self, buf):
        dev = self.decompress_dev(unpack(buf))
        t = Timer(); out = cp.asnumpy(dev); t("download")
        self.timing["decompress"].update(t.parts)
        return out


# ====================================================================== M25
class M25BundleGPU:
    name = "M25 model-repo bundle on GPU (port of M23)"

    def __init__(self):
        self.timing = {}
        self.base_codec = M24BF16GPU()

    @staticmethod
    def _q_layout(qdata, base_g):
        g = GGUF.from_bytes(qdata)
        by_name = {t.name: t for t in base_g.tensors}
        out = []
        for t in sorted(g.tensors, key=lambda t: t.offset):
            r = by_name.get(t.name)
            if t.type == GGML_Q8_0 and r is not None and r.type == GGML_BF16 and r.nbytes == t.nbytes // 34 * 64:
                out.append([t.offset, t.nbytes, r.offset])
        return {"size": len(qdata), "t": out}

    @staticmethod
    def _q_tables(layout):
        tq = np.zeros(len(layout["t"]), TQ_DT); segs = []
        for i, (qoff, nbytes, boff) in enumerate(layout["t"]):
            nb = nbytes // 34
            tq[i] = (qoff, boff, nb, i * TPQ)
            for b0 in range(0, nb, SEGQ):
                segs.append((i, b0, min(SEGQ, nb - b0), 0))
        return tq, np.array(segs, SQ_DT)

    @staticmethod
    def _rest(qdata, layout):
        parts, pos = [], 0
        for off, nbytes, _ in layout["t"]:
            parts.append(qdata[pos:off]); pos = off + nbytes
        parts.append(qdata[pos:])
        return b"".join(parts)

    def compress_files(self, base: bytes, others: list) -> bytes:
        ev = Timer()
        base_dev = cp.asarray(np.frombuffer(base, np.uint8))
        ch = {"base": pack(self.base_codec.compress_dev(base, base_dev))}
        ev("base (M24)")
        base_g = GGUF.from_bytes(base)
        zdict = zstandard.ZstdCompressionDict(base[:base_g.data_start], dict_type=zstandard.DICT_TYPE_RAWCONTENT)
        pool = ThreadPoolExecutor(2)
        for i, q in enumerate(others):
            layout = self._q_layout(q, base_g)
            rest_job = pool.submit(lambda q=q, layout=layout: zstandard.ZstdCompressor(level=19, dict_data=zdict).compress(self._rest(q, layout)))
            qdev = cp.asarray(np.frombuffer(q, np.uint8))
            tq, sq = self._q_tables(layout)
            args = (qdev, base_dev, cp.asarray(tq.view(np.uint8)), cp.asarray(sq.view(np.uint8)), np.int32(len(sq)))
            counts = cp.zeros((len(tq) * TPQ, 256), cp.uint32)
            kern("q_hist")(((len(sq) + 63) // 64,), (64,), args + (counts,))
            stream, lens, tables = _rans_encode("q_enc", len(sq), args, counts, SEGQ * 34 * 2 + 1024)
            ch[f"q{i}"] = pack({"layout": zstd_c(json.dumps(layout).encode()), "tables": tables,
                                "lens": zstd_c(lens.astype(np.uint32).tobytes()), "stream": stream,
                                "rest": rest_job.result()})
            del qdev
        ev("quants")
        self.timing["compress"] = ev.parts
        return pack(ch)

    def decompress_files(self, buf: bytes):
        ev = Timer()
        ch = unpack(buf)
        base_dev = self.base_codec.decompress_dev(unpack(ch["base"]))
        ev("base (M24)")
        base_host = cp.asnumpy(base_dev)
        base_g = GGUF.from_bytes(base_host[:64 << 20].tobytes())       # only the header is parsed
        zdict = zstandard.ZstdCompressionDict(base_host[:base_g.data_start].tobytes(), dict_type=zstandard.DICT_TYPE_RAWCONTENT)
        ev("base download")
        outs = []
        for k in sorted(k for k in ch if k.startswith("q")):
            qc = unpack(ch[k])
            layout = json.loads(zstd_d(qc["layout"]))
            rest = zstandard.ZstdDecompressor(dict_data=zdict).decompress(qc["rest"], max_output_size=1 << 32)
            tq, sq = self._q_tables(layout)
            remap, freq, cum, lut = _rans_tables(qc["tables"], len(tq) * TPQ)
            lens = np.frombuffer(zstd_d(qc["lens"]), np.uint32)
            qdev = cp.empty(layout["size"], cp.uint8)
            kern("q_dec")(((len(sq) + 63) // 64,), (64,),
                          (cp.asarray(np.frombuffer(qc["stream"], np.uint8)), _offsets(lens), qdev, base_dev,
                           cp.asarray(tq.view(np.uint8)), cp.asarray(sq.view(np.uint8)), np.int32(len(sq)),
                           remap, freq, cum, lut))
            out = cp.asnumpy(qdev)
            pos = rpos = 0
            for off, nbytes, _ in layout["t"]:
                gap = off - pos
                out[pos:off] = np.frombuffer(rest, np.uint8, gap, rpos); rpos += gap; pos = off + nbytes
            out[pos:] = np.frombuffer(rest, np.uint8, len(rest) - rpos, rpos)
            outs.append(out)
        ev("quants")
        self.timing["decompress"] = ev.parts
        return base_host, outs
