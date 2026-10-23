"""Methods 7-11: second-generation GPU codec (parent: M6).

Engine changes vs M6 (speed):
  * the whole file is uploaded once and Q8_0 blocks are gathered on the GPU;
  * every statistic (scales, column factors, row means, ...) is computed with a few
    vectorised CuPy ops or inside the coding kernels, no sorting / quantiles;
  * two-phase coding: phase A = per-block fields (scale hi/lo, max position, max value),
    phase B = the 31 remaining weights. Phase B contexts may use the row statistics
    that phase A made available to the decoder;
  * the header is xz-compressed in 4 parallel chunks on CPU threads while the GPU works;
  * decompression assembles the output file on the GPU and downloads it once.

Model features (each method enables one more):
  M7  PRED : weight context = expected log-magnitude from a rank-1 variance model
             E|q| ~ colfactor[j] * rowscale[r] / d_block  (64 bins of 1/8 octave)
  M8  SIGN : magnitude | PRED and sign | (PRED, column sign-bias class) coded separately
  M9  RANK : max position coded as its rank among the block's columns ordered by colfactor;
             scale hi byte context += block-column factor
  M10 DEDUP: rows identical to an earlier row of the same tensor are sent as references
  M11 TSEL : per tensor choice between the group tables and its own tables
"""
import json
import lzma
import struct
from concurrent.futures import ThreadPoolExecutor

import cupy as cp
import numpy as np

from codecs_gguf import pack, unpack, zstd_c, zstd_d, group_of
from gguf_reader import GGUF, GGML_Q8_0

PB = 12
SEGA = 4096            # blocks per thread, phase A
SEGB = 1024            # blocks per thread, phase B
NBIN = 64
HEADER_CHUNKS = 4
STAT_ELEMS = 32 << 20  # weights per slice in the per-column statistics

KERNELS = r"""
#define PB 12
#define PSCALE (1u << PB)
#define RANS_L (1u << 23)
#define SEGA %(SEGA)d
#define SEGB %(SEGB)d
#define SIGN %(SIGN)d
#define RANK %(RANK)d
#define HICF %(HICF)d
#define A_HI 0
#define A_LO (256 * (HICF ? 16 : 1))
#define A_POS (A_LO + 256)
#define A_MX (A_POS + 16)
#define TA (A_MX + 16)
#define Q_SG 64
#define TQ (64 + (SIGN ? 1024 : 0))

struct Meta { int first, nblocks, nb_row, grp, colbase, rowbase, off, qset; };

__device__ __forceinline__ int tensor_of(const Meta* M, int ntens, int b) {
    int lo = 0, hi = ntens - 1;
    while (lo < hi) { int m = (lo + hi + 1) >> 1; if (M[m].first <= b) lo = m; else hi = m - 1; }
    return lo;
}

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

#define HIST(tab, s) atomicAdd(&counts[(size_t)(tab) * 256 + (s)], 1u)

#define FLUSH { for (int i_ = 0; i_ < 4; i_++) { *--ptr = (unsigned char)(x & 0xff); x >>= 8; } \
                out_len[s] = (unsigned int)(end - ptr); }
#define LOAD  const unsigned char* ptr = in + in_off[s]; unsigned int x = 0; \
              for (int i_ = 0; i_ < 4; i_++) x = (x << 8) | *ptr++;

// ------------------------------------------------------------------ phase A
#define A_SETUP                                                                       \
    int k = b - M[t].first, inrow = k %% M[t].nb_row;                                   \
    int cb = M[t].colbase / 32 + inrow, c0 = M[t].colbase + inrow * 32;                \
    int base = M[t].grp * TA;

#define A_ARGS const unsigned char* skip, const Meta* M, int ntens, int nblocks,       \
    const unsigned char* bcf, const unsigned char* spread, const unsigned char* perm,  \
    const unsigned char* rankof, const unsigned char* scls

#define B_ARGS const unsigned char* skip, const Meta* M, int ntens, int nblocks,       \
    const short* cfi, const unsigned char* scls, const int* row_m, int qbase, int G

extern "C" __global__ void hist_a(const unsigned char* blocks, const unsigned char* pos, A_ARGS,
                                  unsigned int* counts) {
    int s = blockIdx.x * blockDim.x + threadIdx.x, b0 = s * SEGA;
    if (b0 >= nblocks) return;
    int b1 = min(b0 + SEGA, nblocks), t = tensor_of(M, ntens, b0);
    for (int b = b0; b < b1; b++) {
        while (t + 1 < ntens && b >= M[t + 1].first) t++;
        if (skip[b] & 1) continue;
        A_SETUP
        const unsigned char* blk = blocks + (size_t)b * 34;
        int prev = (inrow == 0 || b == b0) ? 0 : blocks[(size_t)(b - 1) * 34 + 1];
        int hi = blk[1], p = pos[b];
        HIST(base + A_HI + prev + (HICF ? 256 * bcf[cb] : 0), hi);
        HIST(base + A_LO + hi, blk[0]);
        if (!(skip[b] & 4)) {          // bit 4: position / max value are coded by a later phase
            if (RANK) HIST(base + A_POS + spread[cb], rankof[cb * 32 + p]); else HIST(base + A_POS, p);
            HIST(base + A_MX + (SIGN ? scls[c0 + p] : 0), blk[2 + p]);
        }
    }
}

extern "C" __global__ void enc_a(const unsigned char* blocks, const unsigned char* pos, A_ARGS,
                                 const int* remap, const unsigned short* freq, const unsigned int* cum,
                                 unsigned char* out, size_t cap, unsigned int* out_len) {
    int s = blockIdx.x * blockDim.x + threadIdx.x, b0 = s * SEGA;
    if (b0 >= nblocks) return;
    int b1 = min(b0 + SEGA, nblocks), t = tensor_of(M, ntens, b1 - 1);
    unsigned char* end = out + (size_t)(s + 1) * cap; unsigned char* ptr = end;
    unsigned int x = RANS_L;
    for (int b = b1 - 1; b >= b0; b--) {
        while (b < M[t].first) t--;
        if (skip[b] & 1) continue;
        A_SETUP
        const unsigned char* blk = blocks + (size_t)b * 34;
        int prev = (inrow == 0 || b == b0) ? 0 : blocks[(size_t)(b - 1) * 34 + 1];
        int hi = blk[1], p = pos[b];
        if (!(skip[b] & 4)) {
            PUT(base + A_MX + (SIGN ? scls[c0 + p] : 0), blk[2 + p]);
            if (RANK) PUT(base + A_POS + spread[cb], rankof[cb * 32 + p]) else PUT(base + A_POS, p);
        }
        PUT(base + A_LO + hi, blk[0]);
        PUT(base + A_HI + prev + (HICF ? 256 * bcf[cb] : 0), hi);
    }
    FLUSH
}

extern "C" __global__ void dec_a(const unsigned char* in, const unsigned long long* in_off, A_ARGS,
                                 const int* remap, const unsigned short* freq, const unsigned int* cum,
                                 const unsigned char* lut, unsigned char* blocks, unsigned char* pos) {
    int s = blockIdx.x * blockDim.x + threadIdx.x, b0 = s * SEGA;
    if (b0 >= nblocks) return;
    int b1 = min(b0 + SEGA, nblocks), t = tensor_of(M, ntens, b0);
    LOAD
    int prev = 0;
    for (int b = b0; b < b1; b++) {
        while (t + 1 < ntens && b >= M[t + 1].first) t++;
        if (skip[b] & 1) continue;
        A_SETUP
        unsigned char* blk = blocks + (size_t)b * 34;
        if (inrow == 0 || b == b0) prev = 0;
        int hi, lo, p, mx;
        GET(base + A_HI + prev + (HICF ? 256 * bcf[cb] : 0), hi);
        GET(base + A_LO + hi, lo);
        blk[0] = lo; blk[1] = hi; prev = hi;
        if (!(skip[b] & 4)) {
            if (RANK) { int r; GET(base + A_POS + spread[cb], r); p = perm[cb * 32 + r]; }
            else GET(base + A_POS, p);
            GET(base + A_MX + (SIGN ? scls[c0 + p] : 0), mx);
            blk[2 + p] = mx; pos[b] = p;
        }
    }
}

// ------------------------------------------------------------------ phase B
#define B_SETUP                                                                       \
    int k = b - M[t].first, inrow = k %% M[t].nb_row;                                   \
    int c0 = M[t].colbase + inrow * 32;                                                \
    int rm = row_m[M[t].rowbase + k / M[t].nb_row];                                    \
    int bits = blk[0] | (blk[1] << 8);                                                 \
    int sh = rm - bits - M[t].off, p = pos[b];

__device__ __forceinline__ int pbin(int cf, int sh) {
    int u = (cf * 128 + sh) >> 7;
    return u < 0 ? 0 : (u > 63 ? 63 : u);
}

extern "C" __global__ void hist_b(const unsigned char* blocks, const unsigned char* pos, B_ARGS,
                                  unsigned int* counts) {
    int s = blockIdx.x * blockDim.x + threadIdx.x, b0 = s * SEGB;
    if (b0 >= nblocks) return;
    int b1 = min(b0 + SEGB, nblocks), t = tensor_of(M, ntens, b0);
    for (int b = b0; b < b1; b++) {
        while (t + 1 < ntens && b >= M[t + 1].first) t++;
        if (skip[b]) continue;
        const unsigned char* blk = blocks + (size_t)b * 34;
        B_SETUP
        int qb = qbase + (G + t) * TQ;          // histogram per tensor; tables are merged on the host
        for (int j = 0; j < 32; j++) {
            if (j == p) continue;
            int c = c0 + j, pb = pbin(cfi[c], sh), v = (signed char)blk[2 + j];
            if (SIGN) {
                int mag = v < 0 ? -v : v;
                HIST(qb + pb, mag);
                if (mag) HIST(qb + Q_SG + pb * 16 + scls[c], v < 0);
            } else HIST(qb + pb, v + 128);
        }
    }
}

extern "C" __global__ void enc_b(const unsigned char* blocks, const unsigned char* pos, B_ARGS,
                                 const int* remap, const unsigned short* freq, const unsigned int* cum,
                                 unsigned char* out, size_t cap, unsigned int* out_len) {
    int s = blockIdx.x * blockDim.x + threadIdx.x, b0 = s * SEGB;
    if (b0 >= nblocks) return;
    int b1 = min(b0 + SEGB, nblocks), t = tensor_of(M, ntens, b1 - 1);
    unsigned char* end = out + (size_t)(s + 1) * cap; unsigned char* ptr = end;
    unsigned int x = RANS_L;
    for (int b = b1 - 1; b >= b0; b--) {
        while (b < M[t].first) t--;
        if (skip[b]) continue;
        const unsigned char* blk = blocks + (size_t)b * 34;
        B_SETUP
        int qb = qbase + M[t].qset * TQ;
        for (int j = 31; j >= 0; j--) {
            if (j == p) continue;
            int c = c0 + j, pb = pbin(cfi[c], sh), v = (signed char)blk[2 + j];
            if (SIGN) {
                int mag = v < 0 ? -v : v;
                if (mag) PUT(qb + Q_SG + pb * 16 + scls[c], v < 0);
                PUT(qb + pb, mag);
            } else PUT(qb + pb, v + 128);
        }
    }
    FLUSH
}

extern "C" __global__ void dec_b(const unsigned char* in, const unsigned long long* in_off, B_ARGS,
                                 const int* remap, const unsigned short* freq, const unsigned int* cum,
                                 const unsigned char* lut, unsigned char* blocks, const unsigned char* pos) {
    int s = blockIdx.x * blockDim.x + threadIdx.x, b0 = s * SEGB;
    if (b0 >= nblocks) return;
    int b1 = min(b0 + SEGB, nblocks), t = tensor_of(M, ntens, b0);
    LOAD
    for (int b = b0; b < b1; b++) {
        while (t + 1 < ntens && b >= M[t + 1].first) t++;
        if (skip[b]) continue;
        unsigned char* blk = blocks + (size_t)b * 34;
        B_SETUP
        int qb = qbase + M[t].qset * TQ;
        for (int j = 0; j < 32; j++) {
            if (j == p) continue;
            int c = c0 + j, pb = pbin(cfi[c], sh), v;
            if (SIGN) {
                int mag, sg = 0;
                GET(qb + pb, mag);
                if (mag) GET(qb + Q_SG + pb * 16 + scls[c], sg);
                v = sg ? -mag : mag;
            } else { GET(qb + pb, v); v -= 128; }
            blk[2 + j] = (unsigned char)(v & 0xff);
        }
    }
}

// ------------------------------------------------------------------ helpers
extern "C" __global__ void gather(const unsigned char* src, size_t cap, const unsigned int* len,
                                  const unsigned long long* off, int nseg, unsigned char* dst) {
    int s = blockIdx.x;
    if (s >= nseg) return;
    const unsigned char* a = src + (size_t)(s + 1) * cap - len[s];
    unsigned char* d = dst + off[s];
    for (unsigned int i = threadIdx.x; i < len[s]; i += blockDim.x) d[i] = a[i];
}

extern "C" __global__ void build_lut(const unsigned int* cum, int ntab, unsigned char* lut) {
    int tb = blockIdx.x;
    if (tb >= ntab) return;
    for (int sym = threadIdx.x; sym < 256; sym += blockDim.x)
        for (unsigned int i = cum[tb * 257 + sym]; i < cum[tb * 257 + sym + 1]; i++)
            lut[(size_t)tb * PSCALE + i] = (unsigned char)sym;
}
"""

META_DT = np.dtype([(n, "<i4") for n in ("first", "nblocks", "nb_row", "grp", "colbase",
                                         "rowbase", "off", "qset")])


def ta_tq(sign, hicf):
    ta = 256 * (16 if hicf else 1) + 256 + 16 + 16
    return ta, 64 + (1024 if sign else 0)


# ------------------------------------------------------------------ host helpers
def parse_layout(data, with_gguf=False):
    g = GGUF.from_bytes(data)
    q = sorted((t for t in g.tensors if t.type == GGML_Q8_0), key=lambda t: t.offset)
    layout = {"size": len(data), "t": [[t.offset, t.nbytes, t.shape[0], group_of(t.name)] for t in q]}
    return (layout, g) if with_gguf else layout


def rest_of(data, layout):
    parts, pos = [], 0
    for off, nbytes, _, _ in layout["t"]:
        parts.append(data[pos:off]); pos = off + nbytes
    parts.append(data[pos:])
    return b"".join(parts)


def fill_rest(out, rest, layout):
    """the bytes around the Q8_0 tensors (rest_of) back into the file buffer"""
    pos_ = rpos = 0
    for o, nbytes, _, _ in layout["t"]:
        gap = o - pos_
        out[pos_:o] = np.frombuffer(rest, np.uint8, gap, rpos); rpos += gap; pos_ = o + nbytes
    out[pos_:] = np.frombuffer(rest, np.uint8, len(rest) - rpos, rpos)


def xz_chunks_c(buf):
    n = len(buf)
    parts = [buf[i * n // HEADER_CHUNKS:(i + 1) * n // HEADER_CHUNKS] for i in range(HEADER_CHUNKS)]
    with ThreadPoolExecutor(HEADER_CHUNKS) as ex:
        out = list(ex.map(lambda p: lzma.compress(p, preset=9 | lzma.PRESET_EXTREME), parts))
    return struct.pack(f"<{HEADER_CHUNKS}I", *map(len, out)) + b"".join(out)


def xz_chunks_d(buf):
    lens = struct.unpack_from(f"<{HEADER_CHUNKS}I", buf)
    pos, parts = 4 * HEADER_CHUNKS, []
    for ln in lens:
        parts.append(buf[pos:pos + ln]); pos += ln
    with ThreadPoolExecutor(HEADER_CHUNKS) as ex:
        return b"".join(ex.map(lzma.decompress, parts))


# ------------------------------------------------------------------ tokenizer-aware header (M11)
def _items(buf, start):
    """Raw string-array items of a GGUF value span: returns (count, item bytes list, end)."""
    (n,) = struct.unpack_from("<Q", buf, start + 4)
    pos, out = start + 12, []
    for _ in range(n):
        (ln,) = struct.unpack_from("<Q", buf, pos)
        out.append(bytes(buf[pos + 8:pos + 8 + ln])); pos += 8 + ln     # hashable for bytearray input
    return out, start + 12, pos


def _xz(b):
    return lzma.compress(b, preset=9 | lzma.PRESET_EXTREME)


def header_tok_c(data, g, layout):
    """BPE merge k always creates token 256+k, so the merges array is replaced by the
    byte position where each token is split. Tokens are stored as lengths + bytes."""
    rest = rest_of(data, layout)
    if "tokenizer.ggml.merges" not in g.spans or "tokenizer.ggml.tokens" not in g.spans:
        return None
    toks, t0, t1 = _items(data, g.spans["tokenizer.ggml.tokens"][0])
    merges, m0, m1 = _items(data, g.spans["tokenizer.ggml.merges"][0])
    if not (t1 <= m0 and m1 <= g.data_start and len(toks) >= 256 + len(merges)):
        return None
    sp = np.empty(len(merges), np.uint16)
    for k, m in enumerate(merges):
        tok = toks[256 + k]
        p = m.find(b" ")
        if p < 0 or len(m) != len(tok) + 1 or m[:p] != tok[:p] or m[p + 1:] != tok[p:]:
            return None
        sp[k] = p
    lens = np.fromiter(map(len, toks), np.uint32, len(toks))
    if lens.max() >= 65536:
        return None
    stripped = rest[:t0] + rest[t1:m0] + rest[m1:]
    meta = struct.pack("<5Q", t0, t1, m0, m1, len(toks))
    parts = [stripped, lens.astype(np.uint16).tobytes(), b"".join(toks), sp.tobytes()]
    with ThreadPoolExecutor(4) as ex:
        comp = list(ex.map(_xz, parts))
    return meta + struct.pack("<4I", *map(len, comp)) + b"".join(comp)


def header_tok_d(buf):
    t0, t1, m0, m1, ntok = struct.unpack_from("<5Q", buf)
    lens4 = struct.unpack_from("<4I", buf, 40)
    pos, comp = 56, []
    for ln in lens4:
        comp.append(buf[pos:pos + ln]); pos += ln
    with ThreadPoolExecutor(4) as ex:
        stripped, lens, tb, sp = ex.map(lzma.decompress, comp)
    lens = np.frombuffer(lens, np.uint16).astype(np.int64)
    ends = np.cumsum(lens); starts = ends - lens
    toks = [tb[a:b] for a, b in zip(starts.tolist(), ends.tolist())]
    sp = np.frombuffer(sp, np.uint16).tolist()
    tok_items = b"".join(struct.pack("<Q", len(t)) + t for t in toks)
    merge_items = b"".join(struct.pack("<Q", len(toks[256 + k]) + 1) + toks[256 + k][:p] + b" " + toks[256 + k][p:]
                           for k, p in enumerate(sp))
    a = t0
    b = a + (m0 - t1)
    return stripped[:a] + tok_items + stripped[a:b] + merge_items + stripped[b:]


def header_pairs_c(data, g, layout):
    """Token 256+k is the concatenation of the two tokens merged by rule k, so neither the
    merged token strings nor the merges are stored: only the (left id, right id) pairs,
    plus the 256 byte tokens and the trailing special tokens as raw strings."""
    rest = rest_of(data, layout)
    if "tokenizer.ggml.merges" not in g.spans or "tokenizer.ggml.tokens" not in g.spans:
        return None
    toks, t0, t1 = _items(data, g.spans["tokenizer.ggml.tokens"][0])
    merges, m0, m1 = _items(data, g.spans["tokenizer.ggml.merges"][0])
    nm = len(merges)
    if not (t1 <= m0 and m1 <= g.data_start and len(toks) >= 256 + nm):
        return None
    ids = {}
    for i, tk in enumerate(toks):
        ids.setdefault(tk, i)
    pairs = np.empty((nm, 2), np.uint32)
    for k, m in enumerate(merges):
        parts = m.split(b" ")
        if len(parts) != 2 or parts[0] not in ids or parts[1] not in ids:
            return None
        a, b = ids[parts[0]], ids[parts[1]]
        if toks[a] + toks[b] != toks[256 + k]:
            return None
        pairs[k] = (a, b)
    raw_toks = toks[:256] + toks[256 + nm:]
    lens = np.fromiter(map(len, raw_toks), np.uint32, len(raw_toks))
    if lens.max() >= 65536:
        return None
    stripped = rest[:t0] + rest[t1:m0] + rest[m1:]
    # byte planes of the ids compress better than interleaved u32
    planes = np.concatenate([(pairs >> (8 * i)).astype(np.uint8).T.ravel() for i in range(3)])
    meta = struct.pack("<6Q", t0, t1, m0, m1, len(toks), nm)
    parts = [stripped, lens.astype(np.uint16).tobytes(), b"".join(raw_toks), planes.tobytes()]
    with ThreadPoolExecutor(4) as ex:
        comp = list(ex.map(_xz, parts))
    return meta + struct.pack("<4I", *map(len, comp)) + b"".join(comp)


def header_pairs_d(buf):
    t0, t1, m0, m1, ntok, nm = struct.unpack_from("<6Q", buf)
    lens4 = struct.unpack_from("<4I", buf, 48)
    pos, comp = 64, []
    for ln in lens4:
        comp.append(buf[pos:pos + ln]); pos += ln
    with ThreadPoolExecutor(4) as ex:
        stripped, lens, tb, planes = ex.map(lzma.decompress, comp)
    lens = np.frombuffer(lens, np.uint16).astype(np.int64)
    ends = np.cumsum(lens); starts = ends - lens
    raw = [tb[a:b] for a, b in zip(starts.tolist(), ends.tolist())]
    pl = np.frombuffer(planes, np.uint8).reshape(3, 2, nm).astype(np.uint32)
    pairs = (pl[0] | (pl[1] << 8) | (pl[2] << 16)).T.tolist()
    toks = raw[:256]
    merge_items = []
    for a, b in pairs:
        ta, tb_ = toks[a], toks[b]
        toks.append(ta + tb_)
        merge_items.append(struct.pack("<Q", len(ta) + len(tb_) + 1) + ta + b" " + tb_)
    toks.extend(raw[256:])
    tok_items = b"".join(struct.pack("<Q", len(t)) + t for t in toks)
    a = t0
    b = a + (m0 - t1)
    return stripped[:a] + tok_items + stripped[a:b] + b"".join(merge_items) + stripped[b:]


def normalize_gpu(counts, scale=1 << PB):
    """counts (T,256) uint32 on GPU -> uint16 freqs of the used rows + used row ids.
    Guarantees: every symbol with a non-zero count gets freq >= 1, every row sums to `scale`."""
    tot = counts.sum(1, dtype=cp.int64)
    used = cp.nonzero(tot)[0]
    c = counts[used].astype(cp.int64)
    t = tot[used][:, None]
    f = cp.where(c > 0, cp.maximum(1, (c * scale + t // 2) // t), 0)
    rows = cp.arange(len(used))
    for _ in range(64):
        diff = scale - f.sum(1)
        if not bool((diff != 0).any()):
            break
        # surplus -> largest entry; deficit -> take one unit from each of the |diff| largest entries > 1
        am = cp.argmax(f, 1)
        f[rows, am] += cp.maximum(diff, 0)
        need = cp.maximum(-diff, 0)
        rank = cp.argsort(cp.argsort(-f, axis=1), axis=1)
        f = f - ((rank < need[:, None]) & (f > 1)).astype(cp.int64)
    assert bool((f.sum(1) == scale).all()) and bool(((c > 0) <= (f > 0)).all())
    return f.astype(cp.uint16), used.astype(cp.int32)


def cum_of(freq):
    cum = cp.zeros((freq.shape[0], 257), cp.uint32)
    cum[:, 1:] = cp.cumsum(freq.astype(cp.uint32), axis=1)
    return cum


def store_tables(freq, used):
    f = freq.get(); u = used.get().astype(np.int64)
    du = np.diff(u, prepend=0).astype(np.uint32)
    return zstd_c(struct.pack("<I", len(u)) + du.tobytes() + f.tobytes(), 19)


def load_tables(buf):
    raw = zstd_d(buf)
    (n,) = struct.unpack_from("<I", raw)
    used = np.cumsum(np.frombuffer(raw, np.uint32, n, 4).astype(np.int64))
    freq = np.frombuffer(raw, np.uint16, n * 256, 4 + 4 * n).reshape(n, 256)
    return freq, used


def _trim():
    """return CuPy's cached free blocks: the pool otherwise keeps every temporary of a large
    tensor, outgrows the GPU memory and WDDM pages it to system memory (10x slower kernels)"""
    cp.get_default_memory_pool().free_all_blocks()


class Timer:
    def __init__(self):
        import time
        self._t = time.perf_counter; self.t = self._t(); self.parts = {}

    def __call__(self, name):
        cp.cuda.Device().synchronize()
        now = self._t(); self.parts[name] = round(now - self.t, 3); self.t = now


# ================================================================== codec
class GPU2:
    name = "GPU2"
    SIGN = RANK = HICF = DEDUP = TSEL = TOKZ = False

    def __init__(self):
        self.timing = {}
        self._mod = None

    def kern(self, name):
        if self._mod is None:
            src = KERNELS % dict(SEGA=SEGA, SEGB=SEGB, SIGN=int(self.SIGN), RANK=int(self.RANK),
                                 HICF=int(self.HICF))
            self._mod = cp.RawModule(code=src, options=("-std=c++17",))
        return self._mod.get_function(name)

    # ---------------------------------------------------------- structure shared by both sides
    def _meta(self, layout, off, qset):
        groups = sorted({t[3] for t in layout["t"]})
        m = np.zeros(len(layout["t"]), META_DT)
        first = colbase = rowbase = 0
        rows_first, rows_nbr = [], []
        for i, (o, nbytes, rowlen, grp) in enumerate(layout["t"]):
            nb, nbr = nbytes // 34, rowlen // 32
            m[i] = (first, nb, nbr, groups.index(grp), colbase, rowbase, off[i], qset[i])
            rows_first.append(first + np.arange(nb // nbr) * nbr); rows_nbr.append(np.full(nb // nbr, nbr))
            first += nb; colbase += rowlen; rowbase += nb // nbr
        return groups, m, np.concatenate(rows_first), np.concatenate(rows_nbr)

    @staticmethod
    def _row_means(allb, rows_first_g, rows_nbr_g):
        bits = allb[:, 0].astype(cp.int64) | (allb[:, 1].astype(cp.int64) << 8)
        cs = cp.concatenate([cp.zeros(1, cp.int64), cp.cumsum(bits)])
        return ((cs[rows_first_g + rows_nbr_g] - cs[rows_first_g]) // rows_nbr_g).astype(cp.int32)

    def _colblock_side(self, cfi, ncol):
        """Per block-column (32 channels) contexts derived from the column factors."""
        c = cfi.reshape(-1, 32).astype(cp.int32)
        key = -c * 64 + cp.arange(32, dtype=cp.int32)[None, :]          # descending cf, ties by index
        perm = cp.argsort(key, axis=1).astype(cp.uint8)
        rankof = cp.argsort(perm, axis=1).astype(cp.uint8)               # inverse permutation
        spread = cp.clip((c.max(1) - cp.sort(c, axis=1)[:, 16]) // 2, 0, 15).astype(cp.uint8)
        return perm.ravel(), rankof.ravel(), spread

    def _bcf(self, cfi, layout):
        """Block-column factor relative to the tensor row average, quarter-octave bins."""
        out, cb = [], 0
        for _, _, rowlen, _ in layout["t"]:
            c = cfi[cb:cb + rowlen].reshape(-1, 32).astype(cp.int32).max(1)
            out.append(cp.clip((c - int(c.mean())) // 2 + 8, 0, 15).astype(cp.uint8)); cb += rowlen
        return cp.concatenate(out)

    # ---------------------------------------------------------- compress
    def compress(self, data, tensors=None, header=True, parsed=None):
        """tensors: indices of the Q8_0 tensors to code (default all; Q8Parts codes a file in parts);
        header=False leaves the non-Q8_0 bytes out (the caller stores them); parsed: parse_layout's
        (layout, gguf) when the caller has it"""
        ev = Timer()
        pool = ThreadPoolExecutor(1)
        layout, g = parsed or parse_layout(data, with_gguf=True)
        rest_job = pool.submit(self._header_c, data, g, layout) if header else None
        if tensors is not None:
            layout = dict(layout, t=[layout["t"][i] for i in tensors])
        view = np.frombuffer(data, np.uint8)
        n = sum(nb for _, nb, _, _ in layout["t"]) // 34
        allb = cp.empty((n, 34), cp.uint8)            # only the coded tensors go to the GPU
        f = 0
        for o, nb, _, _ in layout["t"]:
            allb[f:f + nb // 34] = cp.asarray(view[o:o + nb]).reshape(-1, 34); f += nb // 34
        ev("upload + gather")

        q = allb[:, 2:].view(cp.int8)
        pos = cp.argmax(cp.abs(q.astype(cp.int16)), axis=1).astype(cp.uint8)
        groups, meta0, rf, rn = self._meta(layout, [0] * len(layout["t"]), [0] * len(layout["t"]))
        rows_first, rows_nbr = cp.asarray(rf), cp.asarray(rn)
        row_m = self._row_means(allb, rows_first, rows_nbr)
        bits = (allb[:, 0].astype(cp.int32) | (allb[:, 1].astype(cp.int32) << 8))

        cfi_l, scls_l, off = [], [], []
        f = rb = 0
        for (o, nbytes, rowlen, grp) in layout["t"]:
            nb = nbytes // 34; nbr = rowlen // 32; nrows = nb // nbr
            s1 = cp.zeros(rowlen, cp.float64); s2 = cp.zeros(rowlen, cp.float64)
            step = max(1, STAT_ELEMS // rowlen)                             # rows per slice: bounded memory
            for r0 in range(0, nrows, step):
                r1 = min(r0 + step, nrows); b0, b1 = f + r0 * nbr, f + r1 * nbr
                # block scale relative to its row mean, as a multiplicative factor
                rel = cp.exp2((bits[b0:b1].reshape(r1 - r0, nbr) - row_m[rb + r0:rb + r1, None]).astype(cp.float32) / 1024)
                w = (q[b0:b1].astype(cp.float32).reshape(r1 - r0, nbr, 32) * rel[:, :, None]).reshape(r1 - r0, rowlen)
                s1 += w.sum(0); s2 += (w * w).sum(0)
                del w, rel
            rb += nrows
            ms = (s2 / nrows).astype(cp.float32)
            cf = cp.log2(cp.sqrt(ms) + 1e-6)
            ci = cp.clip(cp.rint(cf * 8), -32768, 32767).astype(cp.int16)
            cfi_l.append(ci)
            bias = (s1 / nrows).astype(cp.float32) / cp.sqrt(ms + 1e-12)
            scls_l.append(cp.clip(cp.floor(bias * 16) + 8, 0, 15).astype(cp.uint8))
            off.append(int(ci.astype(cp.float32).mean() * 128) - 32 * 128)
            f += nb
            _trim()
        cfi = cp.concatenate(cfi_l); scls = cp.concatenate(scls_l)
        if not self.SIGN:
            scls = cp.zeros_like(scls)
        perm, rankof, spread = self._colblock_side(cfi, len(cfi))
        bcf = self._bcf(cfi, layout)
        skip, dups = self._dedup(allb, layout, meta0, rf, rn) if self.DEDUP else (cp.zeros(n, cp.uint8), np.zeros(0, np.int64))
        env = dict(allb=allb, pos=pos, layout=layout, meta=meta0, skip=skip, cfi=cfi, row_m=row_m, off=off)
        self._prepare_extra(env)          # may mark rows with skip=2 (coded by an extra phase)
        ev("statistics")

        G, T = len(groups), len(layout["t"])
        TA, TQ = ta_tq(self.SIGN, self.HICF)
        qbase = G * TA
        ntab = qbase + (G + T) * TQ
        meta = meta0.copy(); meta["off"] = off
        meta_g = cp.asarray(meta.view(np.uint8))
        counts = cp.zeros((ntab, 256), cp.uint32)
        a_args = (skip, meta_g, np.int32(T), np.int32(n), bcf, spread, perm, rankof, scls)
        b_args = (skip, meta_g, np.int32(T), np.int32(n), cfi, scls, row_m, np.int32(qbase), np.int32(G))
        nsa, nsb = (n + SEGA - 1) // SEGA, (n + SEGB - 1) // SEGB
        ga, gb = ((nsa + 63) // 64,), ((nsb + 63) // 64,)
        self.kern("hist_a")(ga, (64,), (allb, pos) + a_args + (counts,))
        self.kern("hist_b")(gb, (64,), (allb, pos) + b_args + (counts,))
        ev("histograms")

        qset = self._select_tables(counts, meta, G, T, qbase, TQ)
        meta["qset"] = qset
        meta_g = cp.asarray(meta.view(np.uint8))
        a_args = (skip, meta_g) + a_args[2:]
        b_args = (skip, meta_g) + b_args[2:]
        freq, used = normalize_gpu(counts)
        remap = cp.full(ntab, -1, cp.int32); remap[used] = cp.arange(len(used), dtype=cp.int32)
        cum = cum_of(freq)
        ev("tables")

        streams = []
        for kname, nseg, grid, seg, args in (("enc_a", nsa, ga, SEGA, a_args), ("enc_b", nsb, gb, SEGB, b_args)):
            cap = seg * 34 * 2 + 16
            out = cp.empty(nseg * cap, cp.uint8)
            out_len = cp.zeros(nseg, cp.uint32)
            self.kern(kname)(grid, (64,), (allb, pos) + args + (remap, freq, cum, out, np.uint64(cap), out_len))
            offs = cp.zeros(nseg + 1, cp.uint64); offs[1:] = cp.cumsum(out_len.astype(cp.uint64))
            stream = cp.empty(int(offs[-1]), cp.uint8)
            self.kern("gather")((nseg,), (256,), (out, np.uint64(cap), out_len, offs, np.int32(nseg), stream))
            del out
            _trim()
            lens = out_len.get()
            assert lens.max() < 65536
            streams.append((stream.get().tobytes(), lens.astype(np.uint16)))
        ev("encode + download")

        env["meta_g"] = meta_g
        extra = self._encode_extra(env)
        ev("extra phase")

        side = {"off": off, "qset": qset.tolist(), "dups": dups.tolist()}
        ch = {
            "layout": zstd_c(json.dumps(layout).encode()),
            "side": zstd_c(json.dumps(side).encode()),
            "cfi": zstd_c(cfi.get().tobytes()),
            "tables": store_tables(freq, used),
            "lens": zstd_c(streams[0][1].tobytes() + streams[1][1].tobytes()),
            "A": streams[0][0],
            "B": streams[1][0],
        }
        if self.SIGN:
            ch["scls"] = zstd_c(scls.get().tobytes())
        ch.update(extra)
        if rest_job is not None:
            ch["rest"] = rest_job.result()
        ev("pack (wait header)")
        self.timing["compress"] = ev.parts
        return pack(ch)

    def _header_c(self, data, g, layout):
        if getattr(self, "TOKPAIRS", False):
            h = header_pairs_c(data, g, layout)
            if h is not None:
                return b"P" + h
        if self.TOKZ:
            h = header_tok_c(data, g, layout)
            if h is not None:
                return b"T" + h
        return b"X" + xz_chunks_c(rest_of(data, layout))

    @staticmethod
    def _header_d(buf):
        if buf[:1] == b"P":
            return header_pairs_d(buf[1:])
        return header_tok_d(buf[1:]) if buf[:1] == b"T" else xz_chunks_d(buf[1:])

    # ---------------------------------------------------------- extension points (GPU3)
    def _prepare_extra(self, env):
        pass

    def _encode_extra(self, env):
        return {}

    def _mark_extra(self, ch, skip, meta):
        pass

    def _decode_extra(self, ch, env):
        pass

    # ---------------------------------------------------------- per-tensor table selection
    def _select_tables(self, counts, meta, G, T, qbase, TQ):
        qset = meta["grp"].astype(np.int32).copy()
        if not self.TSEL:
            # merge tensor histograms into their group's sets
            for t in range(T):
                g = qset[t]
                counts[qbase + g * TQ: qbase + (g + 1) * TQ] += counts[qbase + (G + t) * TQ: qbase + (G + t + 1) * TQ]
                counts[qbase + (G + t) * TQ: qbase + (G + t + 1) * TQ] = 0
            return qset
        ct = counts[qbase + G * TQ:].reshape(T, TQ, 256).astype(cp.float64)
        grp = cp.asarray(meta["grp"])
        cg = cp.zeros((G, TQ, 256), cp.float64)
        cp.add.at(cg, grp, ct)

        def bits_under(c, p_counts):
            tot = p_counts.sum(-1, keepdims=True)
            lp = cp.where(p_counts > 0, cp.log2(p_counts / cp.maximum(tot, 1)), 0)
            return -(c * lp).sum((-1, -2))
        own = bits_under(ct, ct)
        shared = bits_under(ct, cg[grp])
        table_bits = (ct > 0).sum((-1, -2)) * 12.0          # ~1.5 bytes per stored frequency
        choose = ((shared - own) > table_bits).get()
        for t in range(T):
            g = int(meta["grp"][t])
            if choose[t]:
                qset[t] = G + t
            else:
                counts[qbase + g * TQ: qbase + (g + 1) * TQ] += counts[qbase + (G + t) * TQ: qbase + (G + t + 1) * TQ]
                counts[qbase + (G + t) * TQ: qbase + (G + t + 1) * TQ] = 0
        return qset

    # ---------------------------------------------------------- duplicate rows
    def _dedup(self, allb, layout, meta, rf, rn):
        n = allb.shape[0]
        skip = cp.zeros(n, cp.uint8)
        dups = []
        rng = cp.random.RandomState(1234)
        for i, (o, nbytes, rowlen, grp) in enumerate(layout["t"]):
            nbr = rowlen // 32
            f = int(meta["first"][i]); nb = int(meta["nblocks"][i])
            rows = allb[f:f + nb].reshape(nb // nbr, nbr * 34).view(cp.uint64)
            mult = rng.randint(1, 2**62, size=rows.shape[1], dtype=cp.int64).astype(cp.uint64) | cp.uint64(1)
            h = (rows * mult[None, :]).sum(1).get()
            _, first_idx, inv = np.unique(h, return_index=True, return_inverse=True)
            src = first_idx[inv]
            cand = np.nonzero(src != np.arange(len(h)))[0]
            if not len(cand):
                continue
            eq = (rows[cp.asarray(cand)] == rows[cp.asarray(src[cand])]).all(1).get()
            for r, s in zip(cand[eq], src[cand][eq]):
                dups.append((i, int(r), int(s)))
                skip[f + r * nbr: f + (r + 1) * nbr] = 1
        return skip, np.array(dups, np.int64).reshape(-1, 3)

    # ---------------------------------------------------------- decompress
    def decompress(self, buf, out=None):
        """-> numpy uint8 buffer with the exact file bytes; with `out` (a part of Q8Parts) only the
        part's tensors are written into it"""
        ev = Timer()
        ch = unpack(buf)
        rest_job = ThreadPoolExecutor(1).submit(self._header_d, ch["rest"]) if "rest" in ch else None
        layout = json.loads(zstd_d(ch["layout"]))
        side = json.loads(zstd_d(ch["side"]))
        groups, meta, rf, rn = self._meta(layout, side["off"], side["qset"])
        G, T = len(groups), len(layout["t"])
        TA, TQ = ta_tq(self.SIGN, self.HICF)
        qbase = G * TA
        ntab = qbase + (G + T) * TQ
        n = int(meta["first"][-1] + meta["nblocks"][-1])
        nsa, nsb = (n + SEGA - 1) // SEGA, (n + SEGB - 1) // SEGB
        lens = np.frombuffer(zstd_d(ch["lens"]), np.uint16).astype(np.uint64)
        freq_h, used_h = load_tables(ch["tables"])
        cfi = cp.asarray(np.frombuffer(zstd_d(ch["cfi"]), np.int16))
        scls = cp.asarray(np.frombuffer(zstd_d(ch["scls"]), np.uint8)) if self.SIGN else cp.zeros(len(cfi), cp.uint8)
        ev("unpack")

        freq = cp.asarray(freq_h); cum = cum_of(freq)
        lut = cp.empty((len(freq_h), 1 << PB), cp.uint8)
        self.kern("build_lut")((len(freq_h),), (256,), (cum, np.int32(len(freq_h)), lut))
        remap = cp.full(ntab, -1, cp.int32); remap[cp.asarray(used_h)] = cp.arange(len(used_h), dtype=cp.int32)
        perm, rankof, spread = self._colblock_side(cfi, len(cfi))
        bcf = self._bcf(cfi, layout)
        skip = cp.zeros(n, cp.uint8)
        dups = np.array(side["dups"], np.int64).reshape(-1, 3)
        for i, r, _ in dups:
            nbr = int(meta["nb_row"][i]); f = int(meta["first"][i])
            skip[f + r * nbr: f + (r + 1) * nbr] = 1
        self._mark_extra(ch, skip, meta)
        meta_g = cp.asarray(meta.view(np.uint8))
        allb = cp.zeros((n, 34), cp.uint8)
        pos = cp.zeros(n, cp.uint8)
        sa = cp.asarray(np.frombuffer(ch["A"], np.uint8)); sb = cp.asarray(np.frombuffer(ch["B"], np.uint8))
        oa = np.zeros(nsa, np.uint64); oa[1:] = np.cumsum(lens[:nsa])[:-1]
        ob = np.zeros(nsb, np.uint64); ob[1:] = np.cumsum(lens[nsa:])[:-1]
        ev("upload + luts")

        self.kern("dec_a")(((nsa + 63) // 64,), (64,),
                           (sa, cp.asarray(oa), skip, meta_g, np.int32(T), np.int32(n), bcf, spread, perm, rankof, scls,
                            remap, freq, cum, lut, allb, pos))
        row_m = self._row_means(allb, cp.asarray(rf), cp.asarray(rn))
        self.kern("dec_b")(((nsb + 63) // 64,), (64,),
                           (sb, cp.asarray(ob), skip, meta_g, np.int32(T), np.int32(n), cfi, scls, row_m,
                            np.int32(qbase), np.int32(G), remap, freq, cum, lut, allb, pos))
        self._decode_extra(ch, dict(allb=allb, pos=pos, layout=layout, meta=meta, meta_g=meta_g, skip=skip,
                                    cfi=cfi, row_m=row_m, off=side["off"]))
        if len(dups):
            dst, src = [], []
            for i, r, s_ in dups:
                nbr = int(meta["nb_row"][i]); f = int(meta["first"][i])
                dst.append(f + r * nbr + np.arange(nbr)); src.append(f + s_ * nbr + np.arange(nbr))
            allb[cp.asarray(np.concatenate(dst))] = allb[cp.asarray(np.concatenate(src))]
        ev("decode")

        if out is None:
            out = np.empty(layout["size"], np.uint8)
        for (o, nbytes, _, _), f, nb in zip(layout["t"], meta["first"], meta["nblocks"]):
            allb[f:f + nb].ravel().get(out=out[o:o + nbytes])
        del allb
        ev("download")
        if rest_job is not None:
            fill_rest(out, rest_job.result(), layout)
        ev("header")
        self.timing["decompress"] = ev.parts
        return out


class M7Pred(GPU2):
    name = "M7 GPU2 engine + rank-1 variance context"


class M8Sign(GPU2):
    name = "M8 = M7 + sign|column-bias split"
    SIGN = True


class M9Rank(GPU2):
    name = "M9 = M8 + max-position rank + scale col ctx"
    SIGN = RANK = HICF = True


class M10Dedup(GPU2):
    name = "M10 = M9 + duplicate-row references"
    SIGN = RANK = HICF = DEDUP = True


class M11Tokenizer(GPU2):
    name = "M11 = M10 + tokenizer-aware header (merges from vocab)"
    SIGN = RANK = HICF = DEDUP = TOKZ = True


METHODS2 = [M7Pred(), M8Sign(), M9Rank(), M10Dedup(), M11Tokenizer()]
