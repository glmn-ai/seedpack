"""Methods 17-21: GPU3 = GPU2 (M10/M11 model) + inter-row prediction.

Measurements showed the per-weight model is at the i.i.d. entropy bound; the only large
remaining redundancy is *between rows*: token embeddings of related tokens are similar
(nearest earlier row: cosine 0.51 on average). Rows of selected tensors are therefore
predicted from K similar rows decoded earlier:

    w_hat[r, j] = mu[j] + sum_k coef[r, k] * (w[parent_k(r), j] - mu[j])
    pred_q      = clip(rint(w_hat / d_block), -127, 127)      (deterministic fp32 kernel)
    coded       = (q - pred_q) mod 256, context = M11 variance bin shifted by the row's
                  residual energy class

Rows are split into WAVES contiguous groups; parents always come from earlier waves, so
the decoder runs one predict + decode kernel pair per wave.
"""
import json
import struct

import cupy as cp
import numpy as np

from codecs_gguf import pack, unpack, zstd_c, zstd_d
from gpu2 import GPU2, normalize_gpu, cum_of, PB, store_tables, load_tables, _trim, parse_layout, fill_rest

C_KERNELS = r"""
#define PB 12
#define PSCALE (1u << PB)
#define RANS_L (1u << 23)
#define PREDA %(PREDA)d
#define TPC (64 + (PREDA ? 24 : 0))     // tables per predicted tensor: residual, [rank, max value]

struct Meta { int first, nblocks, nb_row, grp, colbase, rowbase, off, qset; };

__device__ __forceinline__ float h2f(unsigned int h) {
    unsigned int s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023, f;
    if (e == 0) {
        if (m == 0) f = s << 31;
        else { int ee = 113; while (!(m & 1024)) { m <<= 1; ee--; } m &= 1023; f = (s << 31) | (ee << 23) | (m << 13); }
    } else if (e == 31) f = (s << 31) | 0x7f800000u | (m << 13);
    else f = (s << 31) | ((e + 112) << 23) | (m << 13);
    return __int_as_float(f);
}

__device__ __forceinline__ int pbin(int cf, int sh) {
    int u = (cf * 128 + sh) >> 7;
    return u < 0 ? 0 : (u > 63 ? 63 : u);
}

// rank of position p among the block |prediction| values (descending, ties by index)
__device__ __forceinline__ int rank_of(const signed char* pr, int p) {
    int ap = abs((int)pr[p]), r = 0;
    for (int k = 0; k < 32; k++) { int a = abs((int)pr[k]); r += (a > ap) || (a == ap && k < p); }
    return r;
}
__device__ __forceinline__ int pos_of_rank(const signed char* pr, int rk) {
    for (int j = 0; j < 32; j++) if (rank_of(pr, j) == rk) return j;
    return 0;
}
__device__ __forceinline__ int rank_ctx(const signed char* pr) {
    int m = 0; for (int k = 0; k < 32; k++) m = max(m, abs((int)pr[k]));
    return min(m / 16, 7);
}
__device__ __forceinline__ int mx_ctx(const signed char* pr, int p) {
    int v = pr[p]; return (v < 0) * 8 + min(abs(v) / 16, 7);
}

// deterministic row prediction (explicit IEEE ops, no FMA contraction)
__device__ __forceinline__ signed char pred1(const unsigned char* blocks, int first, int nbr, int r, int j,
                                             const int* par, const float* coef, int K, const float* mu) {
    float acc = 0.0f;
    for (int k = 0; k < K; k++) {
        int p = par[(size_t)r * K + k];
        if (p < 0) continue;
        const unsigned char* pb = blocks + (size_t)(first + (size_t)p * nbr + j / 32) * 34;
        float w = __fmul_rn((float)(signed char)pb[2 + j %% 32], h2f(pb[0] | (pb[1] << 8)));
        acc = __fadd_rn(acc, __fmul_rn(coef[(size_t)r * K + k], __fsub_rn(w, mu[j])));
    }
    float m = __fadd_rn(acc, mu[j]);
    const unsigned char* tb = blocks + (size_t)(first + (size_t)r * nbr + j / 32) * 34;
    float d = h2f(tb[0] | (tb[1] << 8));
    int v = 0;
    if (d > 0.0f) {
        float t = rintf(__fdiv_rn(m, d));
        v = (int)fmaxf(-127.0f, fminf(127.0f, t));
    }
    return (signed char)v;
}

extern "C" __global__ void predict(const unsigned char* blocks, int first, int nbr, int C,
                                   const int* rows, int nrw, const int* par, const float* coef, int K,
                                   const float* mu, signed char* pint) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)nrw * C) return;
    int r = rows[idx / C], j = (int)(idx %% C);
    pint[(size_t)r * C + j] = pred1(blocks, first, nbr, r, j, par, coef, K, mu);
}

// the rows of one wave of every predicted tensor at once (decoder); entry e = row ent_row[e]
// of slot ent_slot[e], one thread per (entry, column < CMAX)
extern "C" __global__ void predict_multi(const unsigned char* blocks, const int* ent_slot, const int* ent_row,
                                         int nent, int CMAX, const int* s_first, const int* s_nbr,
                                         const long long* s_rbase, const long long* s_pbase, const int* s_mubase,
                                         const int* par, const float* coef, int K, const float* mu,
                                         signed char* pint) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)nent * CMAX) return;
    int e = (int)(idx / CMAX), j = (int)(idx %% CMAX), s = ent_slot[e], r = ent_row[e];
    int nbr = s_nbr[s], C = nbr * 32;
    if (j >= C) return;
    pint[s_pbase[s] + (size_t)r * C + j] = pred1(blocks, s_first[s], nbr, r, j, par + s_rbase[s] * K,
                                                 coef + s_rbase[s] * K, K, mu + s_mubase[s]);
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

// segment s covers rows [seg_r0, seg_r0 + seg_nr) of tensor seg_t (index into M)
#define C_ARGS const unsigned char* skip, const Meta* M, const int* seg_t, const int* seg_r0,         \
    const int* seg_nr, const int* seg_slot, int nseg, const short* cfi, const int* row_m,              \
    const signed char* fbin, const long long* rbase, const signed char* pint, const long long* pbase

#define C_ELEM_SETUP                                                                   \
    int t = seg_t[s], slot = seg_slot[s], nbr = M[t].nb_row, C = nbr * 32;              \
    const signed char* fb = fbin + rbase[slot];                                         \
    const signed char* pi = pint + pbase[slot];                                         \
    int tb = slot * TPC;

extern "C" __global__ void hist_c(const unsigned char* blocks, const unsigned char* pos, C_ARGS,
                                  unsigned int* counts) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nseg) return;
    C_ELEM_SETUP
    for (int r = seg_r0[s]; r < seg_r0[s] + seg_nr[s]; r++)
        for (int kk = 0; kk < nbr; kk++) {
            size_t b = (size_t)M[t].first + (size_t)r * nbr + kk;
            if (skip[b] & 1) continue;
            const unsigned char* blk = blocks + b * 34;
            int sh = row_m[M[t].rowbase + r] - (blk[0] | (blk[1] << 8)) - M[t].off, p = pos[b];
            const signed char* pr = pi + (size_t)r * C + kk * 32;
            if (PREDA) {
                atomicAdd(&counts[(size_t)(tb + 64 + rank_ctx(pr)) * 256 + rank_of(pr, p)], 1u);
                atomicAdd(&counts[(size_t)(tb + 72 + mx_ctx(pr, p)) * 256 + blk[2 + p]], 1u);
            }
            for (int j = 0; j < 32; j++) {
                if (j == p) continue;
                int c = M[t].colbase + kk * 32 + j;
                int ctx = pbin(cfi[c], sh) + fb[r]; ctx = ctx < 0 ? 0 : (ctx > 63 ? 63 : ctx);
                int sym = ((signed char)blk[2 + j] - pr[j]) & 255;
                atomicAdd(&counts[(size_t)(tb + ctx) * 256 + sym], 1u);
            }
        }
}

extern "C" __global__ void enc_c(const unsigned char* blocks, const unsigned char* pos, C_ARGS,
                                 const int* remap, const unsigned short* freq, const unsigned int* cum,
                                 unsigned char* out, size_t cap, unsigned int* out_len) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nseg) return;
    C_ELEM_SETUP
    unsigned char* end = out + (size_t)(s + 1) * cap; unsigned char* ptr = end;
    unsigned int x = RANS_L;
    for (int r = seg_r0[s] + seg_nr[s] - 1; r >= seg_r0[s]; r--)
        for (int kk = nbr - 1; kk >= 0; kk--) {
            size_t b = (size_t)M[t].first + (size_t)r * nbr + kk;
            if (skip[b] & 1) continue;
            const unsigned char* blk = blocks + b * 34;
            int sh = row_m[M[t].rowbase + r] - (blk[0] | (blk[1] << 8)) - M[t].off, p = pos[b];
            const signed char* pr = pi + (size_t)r * C + kk * 32;
            for (int j = 31; j >= 0; j--) {
                if (j == p) continue;
                int c = M[t].colbase + kk * 32 + j;
                int ctx = pbin(cfi[c], sh) + fb[r]; ctx = ctx < 0 ? 0 : (ctx > 63 ? 63 : ctx);
                int sym = ((signed char)blk[2 + j] - pr[j]) & 255;
                PUT(tb + ctx, sym);
            }
            if (PREDA) {
                PUT(tb + 72 + mx_ctx(pr, p), blk[2 + p]);
                PUT(tb + 64 + rank_ctx(pr), rank_of(pr, p));
            }
        }
    for (int i = 0; i < 4; i++) { *--ptr = (unsigned char)(x & 0xff); x >>= 8; }
    out_len[s] = (unsigned int)(end - ptr);
}

extern "C" __global__ void dec_c(const unsigned char* in, const unsigned long long* in_off, int seg0,
                                 unsigned char* pos, C_ARGS,
                                 const int* remap, const unsigned short* freq, const unsigned int* cum,
                                 const unsigned char* lut, unsigned char* blocks) {
    int s = seg0 + blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nseg) return;
    C_ELEM_SETUP
    const unsigned char* ptr = in + in_off[s]; unsigned int x = 0;
    for (int i = 0; i < 4; i++) x = (x << 8) | *ptr++;
    for (int r = seg_r0[s]; r < seg_r0[s] + seg_nr[s]; r++)
        for (int kk = 0; kk < nbr; kk++) {
            size_t b = (size_t)M[t].first + (size_t)r * nbr + kk;
            if (skip[b] & 1) continue;
            unsigned char* blk = blocks + b * 34;
            int sh = row_m[M[t].rowbase + r] - (blk[0] | (blk[1] << 8)) - M[t].off, p;
            const signed char* pr = pi + (size_t)r * C + kk * 32;
            if (PREDA) {
                int rk, mx;
                GET(tb + 64 + rank_ctx(pr), rk);
                p = pos_of_rank(pr, rk);
                GET(tb + 72 + mx_ctx(pr, p), mx);
                blk[2 + p] = (unsigned char)mx; pos[b] = (unsigned char)p;
            } else p = pos[b];
            for (int j = 0; j < 32; j++) {
                if (j == p) continue;
                int c = M[t].colbase + kk * 32 + j;
                int ctx = pbin(cfi[c], sh) + fb[r]; ctx = ctx < 0 ? 0 : (ctx > 63 ? 63 : ctx);
                int sym; GET(tb + ctx, sym);
                blk[2 + j] = (unsigned char)((pr[j] + sym) & 255);
            }
        }
}
"""

_cmods = {}
PRED_MEM = 7          # GPU bytes per tensor byte while its rows are planned (f32 rows, fp16 copy, ...)


def ckern(name, preda=False):
    if preda not in _cmods:
        _cmods[preda] = cp.RawModule(code=C_KERNELS % {"PREDA": int(preda)}, options=("-std=c++17", "--fmad=false"))
    return _cmods[preda].get_function(name)


def h2f_np(bits):
    return bits.astype(np.uint16).view(np.float16).astype(np.float32)


class GPU3(GPU2):
    SIGN = RANK = HICF = DEDUP = TOKZ = True
    WAVES = 256            # parents come from earlier waves
    K = 2                  # parents per row
    NCAND = 16             # nearest-neighbour candidates searched
    RPT = 4                # rows per CUDA thread (rANS segment)
    COEF_BITS = 16         # 16 = fp16 coefficients, 8 = int8 fixed point
    ADAPTIVE_K = False     # choose 0..K parents per row by estimated cost
    AUTO_TENSORS = False   # try every tensor, keep where profitable
    PRED_TENSORS = ("token_embd",)
    PREDA = False          # code max position / value of predicted rows from the prediction
    COLRES = False         # phase C column factors from the residual instead of the weights

    # ---------------------------------------------------------- encoder side
    def _row_matrix(self, allb, f, nb, rowlen):
        W = cp.empty((nb * 32,), cp.float32)
        step = 1 << 20                                                # blocks per slice
        for a in range(0, nb, step):
            b = allb[f + a:f + min(a + step, nb)]
            d = h2f_cp(b[:, 0], b[:, 1])
            W[a * 32:(a + len(b)) * 32] = (b[:, 2:].view(cp.int8).astype(cp.float32) * d[:, None]).ravel()
        return W.reshape(-1, rowlen)

    def _quant_coef(self, coef):
        if self.COEF_BITS == 16:
            return coef.astype(cp.float16).astype(cp.float32)
        return cp.clip(cp.rint(coef * 64), -127, 127) / 64

    def _plan_tensor(self, W, valid):
        """kNN in earlier waves + least squares -> parents (n,K), coefs (n,K), fbin (n,), est. gain bits."""
        n, C = W.shape
        P, K, NC = self.WAVES, self.K, self.NCAND
        wave = (cp.arange(n) * P) // n
        wstart = cp.searchsorted(wave, wave, side="left").astype(cp.int32)       # first row of the row's wave
        vf = valid.astype(cp.float32)
        mu = (vf @ W) / max(float(vf.sum()), 1.0)
        col_w = cp.zeros(C, cp.float64)
        for i in range(0, n, 8192):                                   # sum of squares in slices
            col_w += (vf[i:i + 8192] @ (W[i:i + 8192] * W[i:i + 8192])).astype(cp.float64)
        W -= mu                                                       # centred in place
        Wc = W
        Wh = cp.empty((n, C), cp.float16)
        for i in range(0, n, 8192):
            x = Wc[i:i + 8192]
            Wh[i:i + 8192] = (x / cp.maximum(cp.linalg.norm(x, axis=1, keepdims=True), 1e-12)).astype(cp.float16)
        Wh[~valid] = 0
        cand = cp.full((n, NC), -1, cp.int32)
        CH = 1024
        cols = cp.arange(n)
        for i in range(0, n, CH):
            j = min(i + CH, n); rows = cp.arange(j - i)
            if int(wstart[j - 1]) == 0:
                continue
            S = Wh[i:j] @ Wh[:int(wstart[j - 1])].T
            S = cp.where((cols[None, :S.shape[1]] < wstart[i:j, None]) & valid[None, :S.shape[1]], S, cp.float16(-2))
            for k in range(min(NC, K)):             # only the K best candidates become parents
                m = cp.argmax(S, 1)
                ok = S[rows, m] > -1
                cand[i:j, k] = cp.where(ok, m, -1).astype(cp.int32)
                S[rows, m] = -2
        par = cand[:, :K].copy()
        frac = cp.ones(n, cp.float32)
        coefs = cp.zeros((n, K), cp.float32)
        kuse = cp.zeros(n, cp.int32)
        col_res = cp.zeros(C, cp.float64)
        for i in range(0, n, 2048):
            j = min(i + 2048, n)
            y = Wc[i:j]; yy = cp.maximum((y * y).sum(1), 1e-12)
            Xa = Wc[cp.maximum(par[i:j], 0)] * (par[i:j] >= 0)[..., None]
            best = cp.ones(j - i, cp.float32); bestk = cp.zeros(j - i, cp.int32); bestc = cp.zeros((j - i, K), cp.float32)
            ks = range(1, K + 1) if self.ADAPTIVE_K else (K,)
            cost_side = (17 + self.COEF_BITS)
            best_cost = cp.zeros(j - i, cp.float32)                              # cost (bits) of k=0
            res_best = y.copy()
            for k in ks:
                X = Xa[:, :k]
                A = cp.einsum('ckd,cld->ckl', X, X) + 1e-3 * cp.eye(k, dtype=cp.float32)[None]
                c = cp.linalg.solve(A, cp.einsum('ckd,cd->ck', X, y)[..., None])[..., 0]
                c = self._quant_coef(c)
                res = y - cp.einsum('ck,ckd->cd', c, X)
                f = cp.clip((res * res).sum(1) / yy, 1e-3, 4)
                cost = 0.5 * cp.log2(f) * (C - C // 32) + k * cost_side           # relative to no prediction
                better = cost < best_cost
                best_cost = cp.where(better, cost, best_cost)
                best = cp.where(better, f, best); bestk = cp.where(better, k, bestk)
                cpad = cp.zeros((j - i, K), cp.float32); cpad[:, :k] = c
                bestc = cp.where(better[:, None], cpad, bestc)
                res_best = cp.where(better[:, None], res, res_best)
            frac[i:j] = best; kuse[i:j] = bestk; coefs[i:j] = bestc
            vr = valid[i:j, None]
            col_res += cp.where(vr, res_best * res_best, 0).sum(0)
        valid_par = (cp.arange(K)[None, :] < kuse[:, None]) & valid[:, None]
        par = cp.where(valid_par, par, -1).astype(cp.int32)
        coefs = cp.where(valid_par, coefs, 0).astype(cp.float32)
        frac = cp.where(kuse > 0, frac, 1.0)
        if self.COLRES:
            # column factor of the residual (1/8 octave of RMS) and row class relative to the mean
            gfrac = float(col_res.sum() / max(float(col_w.sum()), 1e-30))
            cadj = cp.clip(cp.rint(4 * cp.log2(cp.maximum(col_res / cp.maximum(col_w, 1e-30), 1e-6))), -60, 0).astype(cp.int8)
            fbin = cp.clip(cp.rint(cp.log2(frac / max(gfrac, 1e-6)) * 4), -40, 20).astype(cp.int8)
        else:
            cadj = cp.zeros(C, cp.int8)
            fbin = cp.clip(cp.rint(cp.log2(frac) * 4), -40, 0).astype(cp.int8)
        return par, coefs, fbin, mu.astype(cp.float32), cadj

    def _prepare_extra(self, env):
        allb, layout, meta, skip = env["allb"], env["layout"], env["meta"], env["skip"]
        plans = []
        for i, (o, nbytes, rowlen, grp) in enumerate(layout["t"]):
            if not (self.AUTO_TENSORS or grp in self.PRED_TENSORS):
                continue
            f, nb, nbr = int(meta["first"][i]), int(meta["nblocks"][i]), int(meta["nb_row"][i])
            nrows = nb // nbr
            if nrows < 2 * self.WAVES:
                continue
            if PRED_MEM * nb * 34 > cp.cuda.runtime.memGetInfo()[0] + cp.get_default_memory_pool().free_bytes():
                continue                      # its row matrix would not fit: coded without prediction
            W = self._row_matrix(allb, f, nb, rowlen)
            valid = skip[f:f + nb].reshape(nrows, nbr)[:, 0] == 0
            par, coefs, fbin, mu, cadj = self._plan_tensor(W, valid)
            del W
            _trim()
            plans.append(dict(t=i, par=par, coefs=coefs, fbin=fbin, mu=mu, nrows=nrows, nbr=nbr, first=f, cadj=cadj))
        env["plans"] = plans
        for p in plans:
            sl = skip[p["first"]:p["first"] + p["nrows"] * p["nbr"]]
            sl[...] = cp.where(sl == 1, 1, 6 if self.PREDA else 2).astype(cp.uint8)

    def _waves(self, n):
        wave = (np.arange(n) * self.WAVES) // n
        starts = np.searchsorted(wave, np.arange(self.WAVES))
        return starts, np.append(starts[1:], n)

    def _segments(self, plans):
        """rANS segments ordered (wave, tensor): the decoder handles one wave of all tensors per
        launch -> arrays + wave_start[w] (first segment of wave w, WAVES + 1 entries)"""
        st, sr, sn, ss, wave_start = [], [], [], [], []
        se = [self._waves(p["nrows"]) for p in plans]
        for w in range(self.WAVES):
            wave_start.append(len(st))
            for slot, p in enumerate(plans):
                a, b = se[slot][0][w], se[slot][1][w]
                for r0 in range(a, b, self.RPT):
                    st.append(p["t"]); sr.append(r0); sn.append(min(self.RPT, b - r0)); ss.append(slot)
        wave_start.append(len(st))
        return (np.array(st, np.int32), np.array(sr, np.int32), np.array(sn, np.int32),
                np.array(ss, np.int32), wave_start)

    def _c_args(self, env, plans, segs, fbin_all, rbase, pint_all, pbase):
        st, sr, sn, ss, _ = segs
        cfi_c = env["cfi"].copy()
        meta = env["meta"]
        for p in plans:
            cb = int(meta["colbase"][p["t"]]); C = p["nbr"] * 32
            cfi_c[cb:cb + C] = (cfi_c[cb:cb + C].astype(cp.int32) + p["cadj"].astype(cp.int32)).astype(cp.int16)
        return (env["skip"], env["meta_g"], cp.asarray(st), cp.asarray(sr), cp.asarray(sn), cp.asarray(ss),
                np.int32(len(st)), cfi_c, env["row_m"], fbin_all, cp.asarray(rbase), pint_all, cp.asarray(pbase))

    def _predict(self, env, p, rows, pint_slot):
        C = p["nbr"] * 32
        nr = len(rows)
        if nr == 0:
            return
        total = nr * C
        ckern("predict")(((total + 255) // 256,), (256,),
                         (env["allb"], np.int32(p["first"]), np.int32(p["nbr"]), np.int32(C), rows, np.int32(nr),
                          p["par"], p["coefs"], np.int32(self.K), p["mu"], pint_slot))

    def _encode_extra(self, env):
        plans = env["plans"]
        if not plans:
            return {}
        rbase = np.cumsum([0] + [p["nrows"] for p in plans])[:-1].astype(np.int64)
        pbase = np.cumsum([0] + [p["nrows"] * p["nbr"] * 32 for p in plans])[:-1].astype(np.int64)
        fbin_all = cp.concatenate([p["fbin"] for p in plans])
        pint_all = cp.zeros(int(pbase[-1] + plans[-1]["nrows"] * plans[-1]["nbr"] * 32), cp.int8)
        for slot, p in enumerate(plans):
            n = p["nrows"]; C = p["nbr"] * 32
            self._predict(env, p, cp.arange(n, dtype=cp.int32), pint_all[pbase[slot]:pbase[slot] + n * C])
        segs = self._segments(plans)
        args = self._c_args(env, plans, segs, fbin_all, rbase, pint_all, pbase)
        nseg = len(segs[0])
        counts = cp.zeros((len(plans) * self._tpc(), 256), cp.uint32)
        grid = ((nseg + 63) // 64,)
        ckern("hist_c", self.PREDA)(grid, (64,), (env["allb"], env["pos"]) + args + (counts,))
        freq, used = normalize_gpu(counts)
        remap = cp.full(len(plans) * self._tpc(), -1, cp.int32); remap[used] = cp.arange(len(used), dtype=cp.int32)
        cum = cum_of(freq)
        cap = self.RPT * plans[0]["nbr"] * 34 * 2 + 64
        cap = max(cap, max(self.RPT * p["nbr"] * 34 * 2 + 64 for p in plans))
        out = cp.empty(nseg * cap, cp.uint8); out_len = cp.zeros(nseg, cp.uint32)
        ckern("enc_c", self.PREDA)(grid, (64,), (env["allb"], env["pos"]) + args + (remap, freq, cum, out, np.uint64(cap), out_len))
        lens = out_len.get().astype(np.int64)
        assert lens.max() < 65536
        offs = cp.zeros(nseg + 1, cp.uint64); offs[1:] = cp.cumsum(out_len.astype(cp.uint64))
        dst = cp.empty(int(offs[-1]), cp.uint8)
        self.kern("gather")((nseg,), (256,), (out, np.uint64(cap), out_len, offs, np.int32(nseg), dst))
        stream = dst.get().tobytes()
        del out, dst
        _trim()
        side = {"tensors": [p["t"] for p in plans], "waves": self.WAVES, "K": self.K, "rpt": self.RPT}
        return {
            "C": stream,
            "Clens": zstd_c(lens.astype(np.uint16).tobytes()),
            "Ctables": store_tables(freq, used),
            "Cside": zstd_c(json.dumps(side).encode()),
            "Cpar": zstd_c(cp.concatenate([p["par"].ravel() for p in plans]).get().tobytes(), 19),
            "Ccoef": zstd_c(self._coef_bytes(cp.concatenate([p["coefs"].ravel() for p in plans])), 19),
            "Cfbin": zstd_c(fbin_all.get().tobytes(), 19),
            "Cmu": zstd_c(cp.concatenate([p["mu"] for p in plans]).get().tobytes()),
            "Ccadj": zstd_c(cp.concatenate([p["cadj"] for p in plans]).get().tobytes(), 19),
        }

    def _tpc(self):
        return 64 + (24 if self.PREDA else 0)

    def _coef_bytes(self, c):
        if self.COEF_BITS == 16:
            return c.astype(cp.float16).get().tobytes()
        return cp.rint(c * 64).astype(cp.int8).get().tobytes()

    def _coef_from(self, b):
        if self.COEF_BITS == 16:
            return np.frombuffer(b, np.float16).astype(np.float32)
        return np.frombuffer(b, np.int8).astype(np.float32) / 64

    # ---------------------------------------------------------- decoder side
    def _mark_extra(self, ch, skip, meta):
        if "Cside" not in ch:
            return
        side = json.loads(zstd_d(ch["Cside"]))
        for t in side["tensors"]:
            f, nb = int(meta["first"][t]), int(meta["nblocks"][t])
            sl = skip[f:f + nb]
            sl[...] = cp.where(sl == 1, 1, 6 if self.PREDA else 2).astype(cp.uint8)

    def _decode_extra(self, ch, env):
        if "Cside" not in ch:
            return
        side = json.loads(zstd_d(ch["Cside"]))
        meta = env["meta"]
        K = side["K"]
        par_all = np.frombuffer(zstd_d(ch["Cpar"]), np.int32)
        coef_all = self._coef_from(zstd_d(ch["Ccoef"]))
        fbin_all = cp.asarray(np.frombuffer(zstd_d(ch["Cfbin"]), np.int8))
        mu_all = np.frombuffer(zstd_d(ch["Cmu"]), np.float32)
        cadj_all = np.frombuffer(zstd_d(ch["Ccadj"]), np.int8)
        plans, ro, co = [], 0, 0
        for t in side["tensors"]:
            nbr = int(meta["nb_row"][t]); nrows = int(meta["nblocks"][t]) // nbr; C = nbr * 32
            plans.append(dict(t=t, nrows=nrows, nbr=nbr, first=int(meta["first"][t]),
                              par=cp.asarray(par_all[ro * K:(ro + nrows) * K]),
                              coefs=cp.asarray(coef_all[ro * K:(ro + nrows) * K]),
                              mu=cp.asarray(mu_all[co:co + C]), cadj=cp.asarray(cadj_all[co:co + C])))
            ro += nrows; co += C
        rbase = np.cumsum([0] + [p["nrows"] for p in plans])[:-1].astype(np.int64)
        pbase = np.cumsum([0] + [p["nrows"] * p["nbr"] * 32 for p in plans])[:-1].astype(np.int64)
        pint_all = cp.zeros(int(pbase[-1] + plans[-1]["nrows"] * plans[-1]["nbr"] * 32), cp.int8)
        saved = (self.WAVES, self.K, self.RPT)
        self.WAVES, self.K, self.RPT = side["waves"], K, side["rpt"]
        segs = self._segments(plans)
        args = self._c_args(env, plans, segs, fbin_all, rbase, pint_all, pbase)
        freq_h, used_h = load_tables(ch["Ctables"])
        freq = cp.asarray(freq_h); cum = cum_of(freq)
        lut = cp.empty((len(freq_h), 1 << PB), cp.uint8)
        self.kern("build_lut")((len(freq_h),), (256,), (cum, np.int32(len(freq_h)), lut))
        remap = cp.full(len(plans) * self._tpc(), -1, cp.int32); remap[cp.asarray(used_h)] = cp.arange(len(used_h), dtype=cp.int32)
        lens = np.frombuffer(zstd_d(ch["Clens"]), np.uint16).astype(np.uint64)
        offs = np.zeros(len(lens), np.uint64); offs[1:] = np.cumsum(lens)[:-1]
        stream = cp.asarray(np.frombuffer(ch["C"], np.uint8)); offs_g = cp.asarray(offs)
        wave_start = segs[4]
        # prediction entries (slot, row) ordered by wave, like the segments
        se = [self._waves(p["nrows"]) for p in plans]
        es, er, ent_start = [], [], [0]
        for w in range(self.WAVES):
            for slot in range(len(plans)):
                a, b = se[slot][0][w], se[slot][1][w]
                es.append(np.full(b - a, slot, np.int32)); er.append(np.arange(a, b, dtype=np.int32))
            ent_start.append(ent_start[-1] + sum(len(x) for x in er[-len(plans):]))
        ent_slot, ent_row = cp.asarray(np.concatenate(es)), cp.asarray(np.concatenate(er))
        CMAX = max(p["nbr"] for p in plans) * 32
        mubase = np.cumsum([0] + [p["nbr"] * 32 for p in plans])[:-1].astype(np.int32)
        pargs = (cp.asarray(np.array([p["first"] for p in plans], np.int32)),
                 cp.asarray(np.array([p["nbr"] for p in plans], np.int32)),
                 cp.asarray(rbase), cp.asarray(pbase), cp.asarray(mubase),
                 cp.asarray(par_all), cp.asarray(coef_all), np.int32(K), cp.asarray(mu_all), pint_all)
        for w in range(self.WAVES):
            e0, e1 = ent_start[w], ent_start[w + 1]
            ne = e1 - e0
            ckern("predict_multi")(((ne * CMAX + 255) // 256,), (256,),
                                   (env["allb"], ent_slot[e0:e1], ent_row[e0:e1], np.int32(ne), np.int32(CMAX)) + pargs)
            a, b = wave_start[w], wave_start[w + 1]
            ckern("dec_c", self.PREDA)((((b - a) + 63) // 64,), (64,),
                                       (stream, offs_g, np.int32(a), env["pos"]) + args[:6] + (np.int32(b),) + args[7:] +
                                       (remap, freq, cum, lut, env["allb"]))
        self.WAVES, self.K, self.RPT = saved


def h2f_cp(lo, hi):
    return (lo.astype(cp.uint16) | (hi.astype(cp.uint16) << 8)).view(cp.float16).astype(cp.float32)


class M17RowPred(GPU3):
    name = "M17 = M11 + token_embd row prediction (2 earlier similar rows)"


class M18Compact(GPU3):
    name = "M18 = M17 + int8 coefs, 0..3 parents chosen per row"
    K = 3
    COEF_BITS = 8
    ADAPTIVE_K = True


class M19PredMax(M18Compact):
    name = "M19 = M18 + max position/value of embedding blocks from the prediction"
    PREDA = True


class M20TokPairs(M19PredMax):
    name = "M20 = M19 + tokenizer stored as BPE (left, right) id pairs"
    TOKPAIRS = True


class M21ColRes(M20TokPairs):
    name = "M21 = M20 + residual column factors, tuned waves/parents, attn_q/k rows"
    COLRES = True
    WAVES = 1024
    K = 4
    NCAND = 32
    PRED_TENSORS = ("token_embd", "attn_q", "attn_k")


class M22Fast(M21ColRes):
    name = "M22 = M21 with row prediction for token_embd only, 128 waves (3.7x faster decoding, +0.03%)"
    PRED_TENSORS = ("token_embd",)
    WAVES = 128


METHODS3 = [M17RowPred(), M18Compact(), M19PredMax(), M20TokPairs(), M21ColRes(), M22Fast()]


class Q8Parts:
    """M21 over a file in parts of whole Q8_0 tensors, so GPU memory is bounded by the part size
    (budget bytes; default: a sixth of the free GPU memory) instead of the file size. Each part is
    an independent M21 stream with its own tables; the bytes around the tensors are stored once."""
    MEM = 6              # peak GPU bytes per part byte (blocks + encoder output buffers + streams)

    def __init__(self, budget=None):
        self.c = M22Fast()
        self.budget = budget
        self.parts = 0

    def _budget(self):
        if self.budget:
            return self.budget
        _trim()
        return max(64 << 20, cp.cuda.runtime.memGetInfo()[0] // self.MEM)

    def compress(self, data, with_rest=True):
        """with_rest=False: only the Q8_0 tensors (the caller stores the other bytes)"""
        from concurrent.futures import ThreadPoolExecutor
        layout, g = parse_layout(data, with_gguf=True)
        with ThreadPoolExecutor(1) as ex:
            hdr = ex.submit(self.c._header_c, data, g, layout) if with_rest else None
            budget, groups, cur, size = self._budget(), [], [], 0
            for i, (_, nb, _, _) in enumerate(layout["t"]):
                if cur and size + nb > budget:
                    groups.append(cur); cur, size = [], 0
                cur.append(i); size += nb
            if cur:
                groups.append(cur)
            ch = {"layout": zstd_c(json.dumps(layout).encode())}
            for k, idx in enumerate(groups):
                ch[f"p{k}"] = self.c.compress(data, tensors=idx, header=False, parsed=(layout, g))
                _trim()
            if hdr is not None:
                ch["rest"] = hdr.result()
        self.parts = len(groups)
        return pack(ch)

    def decompress(self, buf, out=None):
        """-> the file (numpy uint8); with `out`, the Q8_0 tensors are written into it"""
        from concurrent.futures import ThreadPoolExecutor
        ch = unpack(buf)
        layout = json.loads(zstd_d(ch["layout"]))
        if out is None:
            out = np.empty(layout["size"], np.uint8)
        with ThreadPoolExecutor(1) as ex:
            rest = ex.submit(self.c._header_d, ch["rest"]) if "rest" in ch else None
            k = 0
            while f"p{k}" in ch:
                self.c.decompress(ch[f"p{k}"], out=out)
                _trim(); k += 1
            if rest is not None:
                fill_rest(out, rest.result(), layout)
        return out
