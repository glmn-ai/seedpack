"""R2: BF16 GGUF codec v2 (candidate successor of M24).

Changes against M24 (src/gpu_bf16.py):
  * 1/8-octave variance side info (rfi/cfi = rint(8*log2 rms)) instead of 1/4 octave;
    exponent context = predicted log2 sigma in 1/8 octave (512 ctx), mantissa context =
    exponent - prediction in 1/8 octave (128 ctx), sign context = column bias class (16 ctx);
  * frequency tables are SHARED by all tensors of the same role (blk.*.ffn_up, ...), which makes
    the finer contexts affordable (table cost ~0.06 MB instead of ~0.3 MB for M24's coarser ones);
  * exact duplicate rows (e.g. untrained token embeddings) are coded once + a row map;
  * bounded GPU memory: tensors are processed in groups of <= group_bytes; the file is kept
    on the GPU between the histogram and the encode pass only when it fits `budget` bytes.
Exactness: contexts are pure integer functions of side info stored in the archive.
"""
import json, os, re, sys, struct
from concurrent.futures import ThreadPoolExecutor

import cupy as cp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from codecs_gguf import pack, unpack, zstd_c, zstd_d
from gguf_reader import GGUF, GGML_BF16
from gpu2 import normalize_gpu, cum_of, store_tables, load_tables, Timer, GPU2
PB = 15                   # probability resolution (M24: 12); +0.25 MB gain, see pb_study.py
from gpu_bf16 import M24BF16GPU

SEG = 16384
K = 8                     # side info resolution: 1/K octave
NE, NM, NS = 512, 128, 16
TPT = NE + NM + NS        # tables per role group

KERNELS = r"""
#define PB %(PB)d
#define PSCALE (1u << PB)
#define RANS_L (1u << 23)

struct TB { long long doff; int rowlen, rowbase, colbase, tab, rmap, pad; };
struct SG { int t; int n; long long i0; };

#define PUT(tab, s) {                                                                 \
    int tt_ = remap[tab]; unsigned int s_ = (s);                                       \
    unsigned int f_ = freq[tt_ * 256 + s_], c_ = cum[tt_ * 257 + s_];                  \
    unsigned int xm_ = ((RANS_L >> PB) << 8) * f_;                                     \
    while (x >= xm_) { *--ptr = (unsigned char)(x & 0xff); x >>= 8; }                  \
    x = ((x / f_) << PB) + (x % f_) + c_; }

#define GET(tab, out) {                                                               \
    int tt_ = remap[tab]; unsigned int slot_ = x & (PSCALE - 1);                       \
    int s_ = lut[(size_t)tt_ * PSCALE + slot_];                                        \
    x = freq[tt_ * 256 + s_] * (x >> PB) + slot_ - cum[tt_ * 257 + s_];                \
    while (x < RANS_L) x = (x << 8) | *ptr++;                                          \
    out = s_; }

__device__ __forceinline__ int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

// compact row rc -> original row
#define OROW(rc) (T.rmap >= 0 ? rowmap[T.rmap + (rc)] : (rc))
#define CTX                                                                            \
    int pred = rfi[T.rowbase + orow] + cfi[T.colbase + j];                             \
    int pe = clampi(pred + 256, 0, 511);
#define MCTX int rl = clampi(8 * ex - pred - 1016 + 64, 0, 127);

extern "C" __global__ void v2_hist(const unsigned char* dev, const TB* tb, const SG* sg, int nseg,
                                   const short* rfi, const short* cfi, const signed char* scl, const int* rowmap,
                                   unsigned int* counts) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nseg) return;
    TB T = tb[sg[s].t];
    const unsigned short* W = (const unsigned short*)(dev + T.doff);
    int rc = (int)(sg[s].i0 / T.rowlen), j = (int)(sg[s].i0 % T.rowlen);
    int orow = OROW(rc);
    const unsigned short* w = W + (size_t)orow * T.rowlen;
    for (int k = 0; k < sg[s].n; k++) {
        unsigned int u = w[j];
        int ex = (u >> 7) & 255, man = u & 127, sgn = u >> 15;
        CTX MCTX
        atomicAdd(&counts[(size_t)(T.tab + pe) * 256 + ex], 1u);
        atomicAdd(&counts[(size_t)(T.tab + 512 + rl) * 256 + man], 1u);
        atomicAdd(&counts[(size_t)(T.tab + 640 + scl[T.colbase + j]) * 256 + sgn], 1u);
        if (++j == T.rowlen) { j = 0; rc++; if (k + 1 < sg[s].n) { orow = OROW(rc); w = W + (size_t)orow * T.rowlen; } }
    }
}

extern "C" __global__ void v2_enc(const unsigned char* dev, const TB* tb, const SG* sg, int nseg,
                                  const short* rfi, const short* cfi, const signed char* scl, const int* rowmap,
                                  const int* remap, const unsigned short* freq, const unsigned int* cum,
                                  unsigned char* out, size_t cap, unsigned int* out_len) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nseg) return;
    TB T = tb[sg[s].t];
    const unsigned short* W = (const unsigned short*)(dev + T.doff);
    long long last = sg[s].i0 + sg[s].n - 1;
    int rc = (int)(last / T.rowlen), j = (int)(last % T.rowlen);
    int orow = OROW(rc);
    const unsigned short* w = W + (size_t)orow * T.rowlen;
    unsigned char* start = out + (size_t)s * cap;
    unsigned char* end = start + cap; unsigned char* ptr = end;
    unsigned int x = RANS_L;
    for (int k = sg[s].n - 1; k >= 0; k--) {
        unsigned int u = w[j];
        int ex = (u >> 7) & 255, man = u & 127, sgn = u >> 15;
        CTX MCTX
        PUT(T.tab + 640 + scl[T.colbase + j], sgn);
        PUT(T.tab + 512 + rl, man);
        PUT(T.tab + pe, ex);
        if (ptr - start < 64) { out_len[s] = 0xffffffffu; return; }
        if (--j < 0) { j = T.rowlen - 1; rc--; if (k > 0) { orow = OROW(rc); w = W + (size_t)orow * T.rowlen; } }
    }
    for (int i = 0; i < 4; i++) { *--ptr = (unsigned char)(x & 0xff); x >>= 8; }
    out_len[s] = (unsigned int)(end - ptr);
}

extern "C" __global__ void v2_dec(const unsigned char* in, const unsigned long long* in_off,
                                  unsigned char* dev, const TB* tb, const SG* sg, int nseg,
                                  const short* rfi, const short* cfi, const signed char* scl, const int* rowmap,
                                  const int* remap, const unsigned short* freq, const unsigned int* cum,
                                  const unsigned char* lut) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nseg) return;
    TB T = tb[sg[s].t];
    unsigned short* W = (unsigned short*)(dev + T.doff);
    int rc = (int)(sg[s].i0 / T.rowlen), j = (int)(sg[s].i0 % T.rowlen);
    int orow = OROW(rc);
    unsigned short* w = W + (size_t)orow * T.rowlen;
    const unsigned char* ptr = in + in_off[s]; unsigned int x = 0;
    for (int i = 0; i < 4; i++) x = (x << 8) | *ptr++;
    for (int k = 0; k < sg[s].n; k++) {
        int ex, man, sgn;
        CTX
        GET(T.tab + pe, ex);
        MCTX
        GET(T.tab + 512 + rl, man);
        GET(T.tab + 640 + scl[T.colbase + j], sgn);
        w[j] = (unsigned short)((sgn << 15) | (ex << 7) | man);
        if (++j == T.rowlen) { j = 0; rc++; if (k + 1 < sg[s].n) { orow = OROW(rc); w = W + (size_t)orow * T.rowlen; } }
    }
}

extern "C" __global__ void build_lut(const unsigned int* cum, int ntab, unsigned char* lut) {
    int tb = blockIdx.x;
    if (tb >= ntab) return;
    for (int sym = threadIdx.x; sym < 256; sym += blockDim.x)
        for (unsigned int i = cum[tb * 257 + sym]; i < cum[tb * 257 + sym + 1]; i++)
            lut[(size_t)tb * PSCALE + i] = (unsigned char)sym;
}

// 64-bit row fingerprints for duplicate-row detection
extern "C" __global__ void row_hash(const unsigned short* w, int nrows, int rowlen, unsigned long long* h) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= nrows) return;
    unsigned long long a = 1469598103934665603ull;
    const unsigned short* p = w + (size_t)r * rowlen;
    for (int j = 0; j < rowlen; j++) { a ^= p[j]; a *= 1099511628211ull; a ^= a >> 29; }
    h[r] = a;
}
""".replace("%(PB)d", str(PB))

TB_DT = np.dtype([("doff", "<i8"), ("rowlen", "<i4"), ("rowbase", "<i4"), ("colbase", "<i4"), ("tab", "<i4"), ("rmap", "<i4"), ("pad", "<i4")])
SG_DT = np.dtype([("t", "<i4"), ("n", "<i4"), ("i0", "<i8")])
_mod = None
_g2 = GPU2()


def kern(name):
    global _mod
    if _mod is None:
        _mod = cp.RawModule(code=KERNELS, options=("-std=c++17",))
    return _mod.get_function(name)


def role(name):
    return re.sub(r"blk\.\d+\.", "", name)


class BF16v2:
    name = "R2 BF16 v2: 1/8-octave contexts, role-shared tables, row dedup, bounded GPU memory"

    def __init__(self, budget=2_000_000_000, group_bytes=400 << 20, dedup=True):
        self.budget, self.group_bytes, self.dedup = budget, group_bytes, dedup
        self.timing = {}

    # ---------------------------------------------------------------- layout
    @staticmethod
    def _layout(g, size):
        ts = sorted((t for t in g.tensors if t.type == GGML_BF16 and len(t.shape) == 2), key=lambda t: t.offset)
        roles = []
        for t in ts:
            if role(t.name) not in roles:
                roles.append(role(t.name))
        return {"size": size, "t": [[t.offset, t.nbytes, t.shape[0], "bf16"] for t in ts],
                "role": [roles.index(role(t.name)) for t in ts], "nroles": len(roles)}

    def _groups(self, layout):
        gs, cur, cb = [], [], 0
        for i, (off, nb, rl, _) in enumerate(layout["t"]):
            if cur and cb + nb > self.group_bytes:
                gs.append(cur); cur, cb = [], 0
            cur.append(i); cb += nb
        if cur:
            gs.append(cur)
        return gs

    @staticmethod
    def _bases(layout):
        rb, cb, rowbase, colbase = [], [], 0, 0
        for off, nb, rl, _ in layout["t"]:
            rb.append(rowbase); cb.append(colbase)
            rowbase += nb // 2 // rl; colbase += rl
        return rb, cb, rowbase, colbase

    def _plan(self, layout, gi, grp, nrows_c, rmap_off):
        """descriptors for one group: TB array (device offsets relative to the group buffer), segments"""
        rb, cb, _, _ = self._bases(layout)
        tb = np.zeros(len(grp), TB_DT); segs = []
        a = layout["t"][grp[0]][0]
        for k, i in enumerate(grp):
            off, nb, rl, _ = layout["t"][i]
            tb[k] = (off - a, rl, rb[i], cb[i], layout["role"][i] * TPT, rmap_off.get(i, -1), 0)
            n = nrows_c[i] * rl
            for i0 in range(0, n, SEG):
                segs.append((k, min(SEG, n - i0), i0))
        return tb, np.array(segs, SG_DT)

    # ---------------------------------------------------------------- compression
    def _stats(self, dev, rl):
        """side info for one tensor in row chunks: rfi (rows), cfi, scl (cols)"""
        u = dev.view(cp.uint16).reshape(-1, rl)
        nr = u.shape[0]; step = max(1, (16 << 20) // rl)
        rf = cp.empty(nr, cp.float64)
        for r0 in range(0, nr, step):
            w = (u[r0:r0 + step].astype(cp.uint32) << 16).view(cp.float32)
            rf[r0:r0 + step] = cp.log2(cp.sqrt((w * w).mean(1, dtype=cp.float64)) + 1e-30)
        rfa = cp.exp2(rf).astype(cp.float32)
        s2 = cp.zeros(rl, cp.float64); s1 = cp.zeros(rl, cp.float64)
        for r0 in range(0, nr, step):
            w = (u[r0:r0 + step].astype(cp.uint32) << 16).view(cp.float32) / rfa[r0:r0 + step, None]
            s2 += (w * w).sum(0, dtype=cp.float64); s1 += w.sum(0, dtype=cp.float64)
        cf = cp.log2(cp.sqrt(s2 / nr) + 1e-30)
        rfi = cp.clip(cp.rint(rf * K), -32768, 32767).astype(cp.int16)
        cfi = cp.clip(cp.rint(cf * K), -32768, 32767).astype(cp.int16)
        bias = (s1 / nr) / cp.exp2(cfi.astype(cp.float64) / K)
        scl = cp.clip(cp.floor(bias * 16) + 8, 0, 15).astype(cp.int8)
        return rfi, cfi, scl

    def _dups(self, dev, rl, host):
        """exact duplicate rows -> (keep rows (sorted), [(dup row, source row)])"""
        u = dev.view(cp.uint16).reshape(-1, rl)
        nr = u.shape[0]
        h = cp.empty(nr, cp.uint64)
        kern("row_hash")(((nr + 127) // 128,), (128,), (u, np.int32(nr), np.int32(rl), h))
        h = h.get()
        order = np.argsort(h, kind="stable")
        hs = h[order]
        same = np.flatnonzero(hs[1:] == hs[:-1])
        if len(same) == 0:
            return None, []
        hu = host.reshape(-1, rl)
        pairs, isdup = [], np.zeros(nr, bool)
        first = {}
        for k in np.unique(np.concatenate([same, same + 1])):
            r = int(order[k]); key = int(hs[k])
            cands = first.setdefault(key, [])
            src = next((c for c in cands if np.array_equal(hu[c], hu[r])), None)
            if src is None:
                cands.append(r)
            else:
                pairs.append((r, src)); isdup[r] = True
        if not pairs:
            return None, []
        return np.flatnonzero(~isdup).astype(np.int32), sorted(pairs)

    def compress(self, data):
        ev = Timer()
        g = GGUF.from_bytes(data)
        layout = self._layout(g, len(data))
        hdr_job = ThreadPoolExecutor(1).submit(M24BF16GPU._header_c, data, g, layout)
        host = np.frombuffer(data, np.uint8)
        groups = self._groups(layout)
        resident = sum(nb for _, nb, _, _ in layout["t"]) + 2 * self.group_bytes < self.budget
        nt = len(layout["t"])
        rfi_l, cfi_l, scl_l = [None] * nt, [None] * nt, [None] * nt
        nrows_c, rowmaps, dup_pairs = {}, {}, {}
        counts = cp.zeros((layout["nroles"] * TPT, 256), cp.uint32)
        devs, plans = {}, {}
        rmap_all = []
        ev("parse")
        # pass 1: statistics, duplicate rows, histograms
        for gi, grp in enumerate(groups):
            a, b = layout["t"][grp[0]][0], layout["t"][grp[-1]][0] + layout["t"][grp[-1]][1]
            dev = cp.asarray(host[a:b])
            rmap_off = {}
            for i in grp:
                off, nb, rl, _ = layout["t"][i]
                d = dev[off - a:off - a + nb]
                rfi_l[i], cfi_l[i], scl_l[i] = self._stats(d, rl)
                keep, pairs = self._dups(d, rl, host[off:off + nb].view(np.uint16)) if self.dedup else (None, [])
                nrows_c[i] = nb // 2 // rl if keep is None else len(keep)
                if keep is not None:
                    rmap_off[i] = sum(len(x) for x in rmap_all); rmap_all.append(keep)
                    dup_pairs[i] = pairs
            rowmaps[gi] = rmap_off
            plans[gi] = self._plan(layout, gi, grp, nrows_c, rmap_off)
            if gi == 0:
                pass
            devs[gi] = (dev, a) if resident else (None, a)
        rfi = cp.concatenate(rfi_l); cfi = cp.concatenate(cfi_l); scl = cp.concatenate(scl_l)
        rowmap = cp.asarray(np.concatenate(rmap_all)) if rmap_all else cp.zeros(1, cp.int32)
        ev("statistics + dedup")
        for gi, grp in enumerate(groups):
            dev, a = devs[gi]
            if dev is None:
                b = layout["t"][grp[-1]][0] + layout["t"][grp[-1]][1]; dev = cp.asarray(host[a:b])
            tb, sg = plans[gi]
            kern("v2_hist")(((len(sg) + 63) // 64,), (64,),
                            (dev, cp.asarray(tb.view(np.uint8)), cp.asarray(sg.view(np.uint8)), np.int32(len(sg)),
                             rfi, cfi, scl, rowmap, counts))
        ev("histogram")
        self.last_counts = counts.get()
        freq, used = normalize_gpu(counts, 1 << PB)
        remap = cp.full(counts.shape[0], -1, cp.int32); remap[used] = cp.arange(len(used), dtype=cp.int32)
        cum = cum_of(freq)
        streams, lens_all = [], []
        cap = SEG * 2 + 1024
        for gi, grp in enumerate(groups):
            dev, a = devs[gi]
            if dev is None:
                b = layout["t"][grp[-1]][0] + layout["t"][grp[-1]][1]; dev = cp.asarray(host[a:b])
            tb, sg = plans[gi]
            out = cp.empty(len(sg) * cap, cp.uint8); out_len = cp.zeros(len(sg), cp.uint32)
            kern("v2_enc")(((len(sg) + 63) // 64,), (64,),
                           (dev, cp.asarray(tb.view(np.uint8)), cp.asarray(sg.view(np.uint8)), np.int32(len(sg)),
                            rfi, cfi, scl, rowmap, remap, freq, cum, out, np.uint64(cap), out_len))
            lens = out_len.get()
            if (lens == 0xffffffff).any():
                raise RuntimeError("rANS segment overflow")
            offs = cp.zeros(len(sg) + 1, cp.uint64); offs[1:] = cp.cumsum(out_len.astype(cp.uint64))
            dst = cp.empty(int(offs[-1]), cp.uint8)
            _g2.kern("gather")((len(sg),), (256,), (out, np.uint64(cap), out_len, offs, np.int32(len(sg)), dst))
            streams.append(dst.get().tobytes()); lens_all.append(lens)
            del out, dst
            devs[gi] = None
        ev("encode")
        dups = [[i, r, s] for i, ps in dup_pairs.items() for r, s in ps]
        meta = dict(layout=layout, dups=dups, nrows_c=[nrows_c[i] for i in range(nt)])
        ch = {
            "meta": zstd_c(json.dumps(meta).encode()),
            "side": zstd_c(rfi.get().tobytes() + cfi.get().tobytes() + scl.get().tobytes(), 19),
            "tables": store_tables(freq, used),
            "lens": zstd_c(np.concatenate(lens_all).astype(np.uint32).tobytes()),
            "stream": b"".join(streams),
            "rest": hdr_job.result(),
        }
        ev("header")
        self.timing["compress"] = ev.parts
        return pack(ch)

    # ---------------------------------------------------------------- decompression
    def decompress(self, buf):
        ev = Timer()
        ch = unpack(buf)
        meta = json.loads(zstd_d(ch["meta"]))
        layout = meta["layout"]; nt = len(layout["t"])
        hdr_job = ThreadPoolExecutor(1).submit(M24BF16GPU._header_d, ch["rest"])
        _, _, nrow, ncol = self._bases(layout)
        side = zstd_d(ch["side"])
        rfi = cp.asarray(np.frombuffer(side, np.int16, nrow, 0))
        cfi = cp.asarray(np.frombuffer(side, np.int16, ncol, 2 * nrow))
        scl = cp.asarray(np.frombuffer(side, np.int8, ncol, 2 * nrow + 2 * ncol))
        freq_h, used_h = load_tables(ch["tables"])
        freq = cp.asarray(freq_h); cum = cum_of(freq)
        lut = cp.empty((len(freq_h), 1 << PB), cp.uint8)
        kern("build_lut")((len(freq_h),), (256,), (cum, np.int32(len(freq_h)), lut))
        remap = cp.full(layout["nroles"] * TPT, -1, cp.int32); remap[cp.asarray(used_h)] = cp.arange(len(used_h), dtype=cp.int32)
        lens = np.frombuffer(zstd_d(ch["lens"]), np.uint32)
        nrows_c = {i: n for i, n in enumerate(meta["nrows_c"])}
        dups = {}
        for i, r, s in meta["dups"]:
            dups.setdefault(i, []).append((r, s))
        # row maps (same construction order as the encoder)
        rmap_all, rmap_off_t = [], {}
        for i in range(nt):
            if i in dups:
                rl = layout["t"][i][2]; nr = layout["t"][i][1] // 2 // rl
                isdup = np.zeros(nr, bool); isdup[[r for r, _ in dups[i]]] = True
                rmap_off_t[i] = sum(len(x) for x in rmap_all); rmap_all.append(np.flatnonzero(~isdup).astype(np.int32))
        rowmap = cp.asarray(np.concatenate(rmap_all)) if rmap_all else cp.zeros(1, cp.int32)
        out = np.empty(layout["size"], np.uint8)
        stream_h = np.frombuffer(ch["stream"], np.uint8)
        ev("unpack")
        spos, lpos = 0, 0
        for gi, grp in enumerate(self._groups(layout)):
            rmap_off = {i: rmap_off_t[i] for i in grp if i in rmap_off_t}
            tb, sg = self._plan(layout, gi, grp, nrows_c, rmap_off)
            gl = lens[lpos:lpos + len(sg)]; lpos += len(sg)
            nbytes = int(gl.sum(dtype=np.uint64))
            offs = np.zeros(len(sg), np.uint64); offs[1:] = np.cumsum(gl.astype(np.uint64))[:-1]
            a = layout["t"][grp[0]][0]; b = layout["t"][grp[-1]][0] + layout["t"][grp[-1]][1]
            dev = cp.empty(b - a, cp.uint8)
            kern("v2_dec")(((len(sg) + 63) // 64,), (64,),
                           (cp.asarray(stream_h[spos:spos + nbytes]), cp.asarray(offs), dev,
                            cp.asarray(tb.view(np.uint8)), cp.asarray(sg.view(np.uint8)), np.int32(len(sg)),
                            rfi, cfi, scl, rowmap, remap, freq, cum, lut))
            spos += nbytes
            for i in grp:
                if i in dups:
                    off, nb, rl, _ = layout["t"][i]
                    u = dev[off - a:off - a + nb].view(cp.uint16).reshape(-1, rl)
                    d = np.array(dups[i], np.int64)
                    u[cp.asarray(d[:, 0])] = u[cp.asarray(d[:, 1])]
            dev.get(out=out[a:b])
            del dev
        ev("decode + download")
        rest = np.frombuffer(hdr_job.result(), np.uint8)
        pos = rpos = 0
        for off, nbytes, _, _ in layout["t"]:
            gap = off - pos
            out[pos:off] = rest[rpos:rpos + gap]; rpos += gap; pos = off + nbytes
        out[pos:] = rest[rpos:]
        ev("header")
        self.timing["decompress"] = ev.parts
        return out
