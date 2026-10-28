"""Method 6: GPU segmented context rANS (CuPy + CUDA kernels compiled with NVRTC).

Same model as M5 (scale bucket x channel class, order-1 scales, max-element
extraction), but the Q8_0 blocks are cut into segments of SEG blocks and every
CUDA thread codes one segment with its own rANS state. All fields of a block
(scale hi, scale lo, max position, max value, 31 weights) go into one interleaved
stream, so a thread can decode its blocks with no dependency on other threads.

Contexts are computed identically inside the hist / encode / decode kernels:
  table = grp * TPG + field_offset + ctx
  hi : ctx = previous hi byte (0 at row start or segment start)      256 tables
  lo : ctx = hi                                                     256 tables
  pos, mx : 1 table each
  qs : ctx = scale_bucket * 16 + column_class                        256 tables
"""
import json
import struct
from concurrent.futures import ThreadPoolExecutor

from .xp import cp
import numpy as np

from . import rans
from .codecs_gguf import (split, join, pack, unpack, zstd_c, zstd_d, xz_c, xz_d,
                         tables_c, tables_d, N_COL, N_SB)

VERBOSE = False
SEG = 512                 # blocks per segment (= per CUDA thread)
PB = 12                   # rANS probability bits on the GPU (4 KB decode LUT per table)
TPG = 256 + 256 + 1 + 1 + N_SB * N_COL      # tables per tensor group
T_HI, T_LO, T_POS, T_MX, T_QS = 0, 256, 512, 513, 514

KERNELS = r"""
#define PB %(PB)d
#define PSCALE (1u << PB)
#define RANS_L (1u << 23)
#define TPG %(TPG)d
#define SEG %(SEG)d

struct Meta {                       // per tensor
    int first;   int nblocks; int nb_row; int grp; int colbase; int med;
};

__device__ __forceinline__ int tensor_of(const int* tfirst, int ntens, int b) {
    int lo = 0, hi = ntens - 1;     // last tensor with first <= b
    while (lo < hi) { int m = (lo + hi + 1) >> 1; if (tfirst[m] <= b) lo = m; else hi = m - 1; }
    return lo;
}

__device__ __forceinline__ int sbucket(int lo, int hi, int med) {
    int bits = lo | (hi << 8);
    int v = ((bits - med) >> 8) + %(NSB2)d;      // arithmetic shift == floor division
    return v < 0 ? 0 : (v > %(NSB1)d ? %(NSB1)d : v);
}

// Walk one block in coding order, calling F(table, symbol). Used by histogram and encoder.
#define BLOCK_SYMBOLS(EMIT)                                                         \
    {                                                                               \
        const unsigned char* blk = blocks + (size_t)b * 34;                         \
        int k = b - M[t].first, inrow = k %% M[t].nb_row;                           \
        int base = M[t].grp * TPG;                                                  \
        int prev = (inrow == 0 || b == b0) ? 0 : blocks[(size_t)(b - 1) * 34 + 1];  \
        int lo = blk[0], hi = blk[1], p = pos[b];                                   \
        EMIT(base + %(T_HI)d + prev, hi);                                           \
        EMIT(base + %(T_LO)d + hi, lo);                                             \
        EMIT(base + %(T_POS)d, p);                                                  \
        EMIT(base + %(T_MX)d, blk[2 + p]);                                          \
        int qb = base + %(T_QS)d + sbucket(lo, hi, M[t].med) * %(NCOL)d;            \
        const unsigned char* cm = colmap + M[t].colbase + inrow * 32;               \
        for (int j = 0; j < 32; j++) if (j != p) EMIT(qb + cm[j], blk[2 + j]);      \
    }

extern "C" __global__ void hist(const unsigned char* blocks, const unsigned char* pos,
                                const Meta* M, const int* tfirst, int ntens,
                                const unsigned char* colmap, int nblocks, unsigned int* counts) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    int b0 = s * SEG;
    if (b0 >= nblocks) return;
    int b1 = min(b0 + SEG, nblocks);
    int t = tensor_of(tfirst, ntens, b0);
    #define H_EMIT(tab, sym) atomicAdd(&counts[(tab) * 256 + (sym)], 1u)
    for (int b = b0; b < b1; b++) {
        if (t + 1 < ntens && b >= tfirst[t + 1]) t++;
        BLOCK_SYMBOLS(H_EMIT)
    }
}

// Encoder: symbols must be pushed in reverse order, so each block's symbols are first
// collected into a small local list (35 entries), then emitted backwards.
extern "C" __global__ void encode(const unsigned char* blocks, const unsigned char* pos,
                                  const Meta* M, const int* tfirst, int ntens,
                                  const unsigned char* colmap, int nblocks,
                                  const unsigned short* freq, const unsigned int* cum,
                                  unsigned char* out, size_t cap, unsigned int* out_len) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    int b0 = s * SEG;
    if (b0 >= nblocks) return;
    int b1 = min(b0 + SEG, nblocks);
    unsigned char* end = out + (size_t)(s + 1) * cap;
    unsigned char* ptr = end;
    unsigned int x = RANS_L;
    int t = tensor_of(tfirst, ntens, b1 - 1);
    int tabs[35]; int syms[35];        // hi, lo, pos, mx + 31 weights
    for (int b = b1 - 1; b >= b0; b--) {
        while (b < M[t].first) t--;
        int n = 0;
        #define E_COLLECT(tab, sym) { tabs[n] = (tab); syms[n] = (sym); n++; }
        BLOCK_SYMBOLS(E_COLLECT)
        for (int i = n - 1; i >= 0; i--) {
            unsigned int f = freq[tabs[i] * 256 + syms[i]];
            unsigned int c = cum[tabs[i] * 257 + syms[i]];
            unsigned int x_max = ((RANS_L >> PB) << 8) * f;
            while (x >= x_max) { *--ptr = (unsigned char)(x & 0xff); x >>= 8; }
            x = ((x / f) << PB) + (x %% f) + c;
        }
    }
    for (int i = 0; i < 4; i++) { *--ptr = (unsigned char)(x & 0xff); x >>= 8; }
    out_len[s] = (unsigned int)(end - ptr);
}

extern "C" __global__ void decode(const unsigned char* in, const unsigned long long* in_off,
                                  const Meta* M, const int* tfirst, int ntens,
                                  const unsigned char* colmap, int nblocks,
                                  const unsigned short* freq, const unsigned int* cum,
                                  const unsigned char* lut, unsigned char* blocks) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    int b0 = s * SEG;
    if (b0 >= nblocks) return;
    int b1 = min(b0 + SEG, nblocks);
    const unsigned char* ptr = in + in_off[s];
    unsigned int x = 0;
    for (int i = 0; i < 4; i++) x = (x << 8) | *ptr++;
    #define D_GET(tab, out) {                                                     \
        unsigned int slot = x & (PSCALE - 1);                                     \
        int sym_ = lut[(size_t)(tab) * PSCALE + slot];                            \
        x = freq[(tab) * 256 + sym_] * (x >> PB) + slot - cum[(tab) * 257 + sym_];\
        while (x < RANS_L) x = (x << 8) | *ptr++;                                 \
        out = sym_; }
    int t = tensor_of(tfirst, ntens, b0);
    int prev = 0;
    for (int b = b0; b < b1; b++) {
        if (t + 1 < ntens && b >= tfirst[t + 1]) t++;
        unsigned char* blk = blocks + (size_t)b * 34;
        int k = b - M[t].first, inrow = k %% M[t].nb_row;
        int base = M[t].grp * TPG;
        if (inrow == 0 || b == b0) prev = 0;
        int hi, lo, p, mx;
        D_GET(base + %(T_HI)d + prev, hi);
        D_GET(base + %(T_LO)d + hi, lo);
        D_GET(base + %(T_POS)d, p);
        D_GET(base + %(T_MX)d, mx);
        blk[0] = lo; blk[1] = hi; blk[2 + p] = mx; prev = hi;
        int qb = base + %(T_QS)d + sbucket(lo, hi, M[t].med) * %(NCOL)d;
        const unsigned char* cm = colmap + M[t].colbase + inrow * 32;
        for (int j = 0; j < 32; j++) if (j != p) { int v; D_GET(qb + cm[j], v); blk[2 + j] = v; }
    }
}

extern "C" __global__ void gather(const unsigned char* src, size_t cap, const unsigned int* len,
                                  const unsigned long long* off, int nseg, unsigned char* dst) {
    // one CUDA block per segment, threads copy bytes
    int s = blockIdx.x;
    if (s >= nseg) return;
    const unsigned char* a = src + (size_t)(s + 1) * cap - len[s];
    unsigned char* d = dst + off[s];
    for (unsigned int i = threadIdx.x; i < len[s]; i += blockDim.x) d[i] = a[i];
}
""" % dict(PB=PB, TPG=TPG, SEG=SEG, NSB2=N_SB // 2, NSB1=N_SB - 1, NCOL=N_COL,
           T_HI=T_HI, T_LO=T_LO, T_POS=T_POS, T_MX=T_MX, T_QS=T_QS)

_mod = None


def kern(name):
    global _mod
    if _mod is None:
        _mod = cp.RawModule(code=KERNELS, options=("-std=c++17",))
    return _mod.get_function(name)


META_DT = np.dtype([("first", "<i4"), ("nblocks", "<i4"), ("nb_row", "<i4"),
                    ("grp", "<i4"), ("colbase", "<i4"), ("med", "<i4")])


def normalize(counts, scale):
    """Frequency normalisation to `scale` (vectorised; rows with no data stay zero)."""
    counts = counts.astype(np.int64)
    tot = counts.sum(1, keepdims=True)
    f = np.where(counts > 0, np.maximum(1, (counts * scale + tot // 2) // np.maximum(tot, 1)), 0)
    for r in np.nonzero(tot.ravel())[0]:
        diff = scale - f[r].sum()
        while diff != 0:              # spread the rounding error over the largest entries
            order = np.argsort(-f[r])
            for j in order:
                if diff == 0:
                    break
                if diff > 0:
                    f[r, j] += 1; diff -= 1
                elif f[r, j] > 1:
                    f[r, j] -= 1; diff += 1
    return f.astype(np.uint16)


def build_luts(freq_g):
    cum = cp.zeros((freq_g.shape[0], 257), cp.uint32)
    cum[:, 1:] = cp.cumsum(freq_g.astype(cp.uint32), axis=1)
    # lut[t, slot] = symbol: searchsorted per table on the cumulative frequencies
    slots = cp.arange(1 << PB, dtype=cp.uint32)
    nt = freq_g.shape[0]
    lut = cp.empty((nt, 1 << PB), cp.uint8)
    step = 256
    for a in range(0, nt, step):
        c = cum[a:a + step, 1:]                                   # (k, 256)
        lut[a:a + step] = (slots[None, :, None] >= c[:, None, :]).sum(2).astype(cp.uint8)
    return cum, lut


class M6GPU:
    name = "M6 GPU segmented context rANS (CUDA)"

    def __init__(self):
        self.timing = {}

    def _meta(self, layout, med):
        groups = sorted({t[3] for t in layout["t"]})
        meta = np.zeros(len(layout["t"]), META_DT)
        first = colbase = 0
        for i, (off, nbytes, rowlen, grp) in enumerate(layout["t"]):
            nb = nbytes // 34
            meta[i] = (first, nb, rowlen // 32, groups.index(grp), colbase, med[i])
            first += nb; colbase += rowlen
        return groups, meta

    def compress(self, data):
        ev = Timer()
        layout, rest, blocks = split(data)
        allb_h = np.concatenate(blocks)
        pool = ThreadPoolExecutor(1)
        rest_job = pool.submit(xz_c, rest)          # header/metadata on CPU while the GPU works
        ev("cpu split")
        allb = cp.asarray(allb_h)
        n = allb.shape[0]
        # per tensor: scale median and channel classes (on GPU)
        med, colmaps, firsts, f = [], [], [], 0
        for (off, nbytes, rowlen, grp) in layout["t"]:
            nb = nbytes // 34
            b = allb[f:f + nb]
            bits = b[:, 0].astype(cp.int32) | (b[:, 1].astype(cp.int32) << 8)
            med.append(int(cp.median(bits)))
            q = b[:, 2:].view(cp.int8).reshape(-1, rowlen).astype(cp.float32)
            rms = cp.sqrt((q * q).mean(0))
            edges = cp.quantile(rms, cp.linspace(0, 1, N_COL + 1)[1:-1])
            colmaps.append(cp.searchsorted(edges, rms).astype(cp.uint8))
            firsts.append(f); f += nb
        colmap = cp.concatenate(colmaps)
        pos = cp.argmax(cp.abs(allb[:, 2:].view(cp.int8).astype(cp.int16)), axis=1).astype(cp.uint8)
        groups, meta = self._meta(layout, med)
        meta_g = cp.asarray(meta.view(np.uint8))
        tfirst = cp.asarray(np.array(firsts, np.int32))
        nseg = (n + SEG - 1) // SEG
        grid, tpb = ((nseg + 127) // 128,), (128,)
        args = (allb, pos, meta_g, tfirst, np.int32(len(firsts)), colmap, np.int32(n))
        cp.cuda.Device().synchronize(); ev("gpu contexts")

        counts = cp.zeros(len(groups) * TPG * 256, cp.uint32)
        kern("hist")(grid, tpb, args + (counts,))
        counts_h = counts.get().reshape(-1, 256)
        ev("gpu histogram")
        freq_h = normalize(counts_h, 1 << PB)
        ev("cpu normalise")
        freq = cp.asarray(freq_h)
        cum, _ = build_luts(freq)

        cap = SEG * 34 * 2 + 16
        out = cp.empty(nseg * cap, cp.uint8)
        out_len = cp.zeros(nseg, cp.uint32)
        kern("encode")(grid, tpb, args + (freq, cum, out, np.uint64(cap), out_len))
        off = cp.zeros(nseg + 1, cp.uint64)
        off[1:] = cp.cumsum(out_len.astype(cp.uint64))
        total = int(off[-1])
        stream = cp.empty(total, cp.uint8)
        kern("gather")((nseg,), (256,), (out, np.uint64(cap), out_len, off, np.int32(nseg), stream))
        stream_h = stream.get()
        lens_h = out_len.get()
        ev("gpu encode")

        assert lens_h.max() < 65536
        used = np.nonzero(counts_h.sum(1))[0].astype(np.uint32)       # store only used tables
        ch = {
            "layout": zstd_c(json.dumps(layout).encode()),
            "rest": rest_job.result(),
            "meta": zstd_c(json.dumps({"groups": groups, "med": med}).encode()),
            "colmap": zstd_c(colmap.get().tobytes()),
            "used": zstd_c(used.tobytes()),
            "tables": tables_c([freq_h[used]]),
            "seglen": zstd_c(lens_h.astype(np.uint16).tobytes()),   # asserted < 65536 below
            "stream": stream_h.tobytes(),
        }
        ev("cpu pack (wait xz header, zstd tables)")
        self.timing["compress"] = ev.parts
        return pack(ch)

    def decompress(self, buf):
        ev = Timer()
        ch = unpack(buf)
        layout = json.loads(zstd_d(ch["layout"]))
        rest_job = ThreadPoolExecutor(1).submit(xz_d, ch["rest"])
        m = json.loads(zstd_d(ch["meta"]))
        groups, meta = self._meta(layout, m["med"])
        used = np.frombuffer(zstd_d(ch["used"]), np.uint32)
        freq_h = np.zeros((len(groups) * TPG, 256), np.uint16)
        freq_h[used] = tables_d(ch["tables"])[0]
        lens = np.frombuffer(zstd_d(ch["seglen"]), np.uint16).astype(np.uint64)
        ev("cpu unpack")
        freq = cp.asarray(freq_h)
        cum, lut = build_luts(freq)
        n = sum(t[1] for t in layout["t"]) // 34
        nseg = len(lens)
        off = np.zeros(nseg, np.uint64); off[1:] = np.cumsum(lens)[:-1]
        stream = cp.asarray(np.frombuffer(ch["stream"], np.uint8))
        colmap = cp.asarray(np.frombuffer(zstd_d(ch["colmap"]), np.uint8))
        tfirst = cp.asarray(meta["first"].astype(np.int32))
        allb = cp.empty((n, 34), cp.uint8)
        cp.cuda.Device().synchronize(); ev("gpu upload + luts")
        kern("decode")(((nseg + 127) // 128,), (128,),
                       (stream, cp.asarray(off), cp.asarray(meta.view(np.uint8)), tfirst,
                        np.int32(len(meta)), colmap, np.int32(n), freq, cum, lut, allb))
        cp.cuda.Device().synchronize(); ev("gpu decode")
        allb_h = allb.get()
        ev("gpu->cpu copy")
        out = join(layout, rest_job.result(), split_blocks(allb_h, layout))
        ev("cpu join")
        self.timing["decompress"] = ev.parts
        return out


def split_blocks(allb, layout):
    out, pos = [], 0
    for t in layout["t"]:
        nb = t[1] // 34
        out.append(allb[pos:pos + nb]); pos += nb
    return out


class Timer:
    def __init__(self):
        import time
        self._time = time.perf_counter
        self.t = self._time(); self.parts = {}

    def __call__(self, name):
        now = self._time(); self.parts[name] = now - self.t; self.t = now
        if VERBOSE: print(f"  [{name}] {self.parts[name]:.2f}s", flush=True)
