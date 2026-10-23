"""Method 26: one archive for a whole model repository (BF16 + every quantisation).

llama.cpp quantisers first choose block scales (possibly with an importance matrix) and then
compute every value deterministically from the original weight, e.g. Q4_K:
    q = clamp(round((w + dmin*m) / (d*sc)), 0, 15)
So once the BF16 weights are in the archive, a quantised file only costs its scales:
every value is replaced by the residual (q - prediction) mod 2^bits *in place* (same bit
layout), which is almost always zero. Scales and metadata are left untouched; float tensors
are XOR-ed with the BF16 tensor converted to their type. The transformed file is then
compressed with zstd using the BF16 header as dictionary. The transform is a bijection given
the BF16 weights, so decompression is exact for any input (a bad prediction only costs size).
Types without a predictor (IQ1/IQ2/IQ3 lattice codes, ...) are stored unchanged.
"""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

import cupy as cp
import numpy as np
import zstandard

from .codecs_gguf import pack, unpack, zstd_c, zstd_d
from .gguf_reader import GGUF
from .gpu_bf16 import M24BF16GPU

F32, F16, BF16 = 0, 1, 30
Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 = 2, 3, 6, 7, 8
Q2_K, Q3_K, Q4_K, Q5_K, Q6_K = 10, 11, 12, 13, 14
IQ4_NL, IQ4_XS = 20, 23
KV_IQ4 = cp.asarray(np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], np.float32))


def h16(b2):
    """(nb,2) uint8 fp16 bytes -> float32 (nb,)"""
    return b2.copy().view(cp.float16).astype(cp.float32).ravel()


def safe_div(a, b):
    return cp.where(b != 0, a / cp.where(b != 0, b, 1), 0)


def roundf(v):                       # C roundf: half away from zero
    return cp.sign(v) * cp.floor(cp.abs(v) + 0.5)


def nearest_iq4(v):
    """ggml best_index_iq4nl over the IQ4 grid (ties as in ggml)."""
    mu = cp.clip(cp.searchsorted(KV_IQ4, v, side="right"), 1, 15)
    lo, hi = KV_IQ4[mu - 1], KV_IQ4[mu]
    idx = cp.where(v - lo < hi - v, mu - 1, mu)
    return cp.where(v <= KV_IQ4[0], 0, cp.where(v >= KV_IQ4[15], 15, idx)).astype(cp.int32)


# ---------------------------------------------------------------- scale fields
# Scales are predicted from the BF16 weights with llama.cpp's own (un-weighted) formulas and
# replaced by (actual - predicted) mod 2^bits. fp16 fields use a monotonic signed ordering.
def h_ord(h):
    h = h.astype(cp.int32)
    return cp.where(h < 0x8000, h, 0x7FFF - h)


def ord_h(o):
    o = o.astype(cp.int32) & 0xFFFF                      # residuals arrive modulo 2^16
    o = cp.where(o >= 0x8000, o - 0x10000, o)
    return cp.where(o >= 0, o, 0x7FFF - o).astype(cp.int32) & 0xFFFF


def f2h_bits(v):
    return v.astype(cp.float16).view(cp.uint16).astype(cp.int32)


def get_h(b, off):
    return (b[:, off].astype(cp.int32) | (b[:, off + 1].astype(cp.int32) << 8))


def set_h(b, off, h):
    b[:, off] = (h & 255).astype(cp.uint8); b[:, off + 1] = ((h >> 8) & 255).astype(cp.uint8)


def signed_max(x):
    """value with the largest magnitude along the last axis (sign kept)"""
    i = cp.argmax(cp.abs(x), -1)
    return cp.take_along_axis(x, i[..., None], -1)[..., 0]


def kmin_scales(x, nsub, nmax):
    """K-quant with min: per sub-block scale=(max-min)/nmax, min=-min(min,0); super d, dmin over /63."""
    xs = x.reshape(len(x), nsub, -1)
    mn = cp.minimum(xs.min(2), 0); mx = xs.max(2)
    sc = (mx - mn) / cp.float32(nmax); m = -mn
    return sc, m


def pack_scale_min_k4(sc, mn):
    """inverse of get_scale_min_k4: 8 six-bit scales + 8 six-bit mins -> 12 bytes"""
    out = cp.zeros((len(sc), 12), cp.int32)
    out[:, 0:4] = (sc[:, 0:4] & 63) | ((sc[:, 4:8] >> 4) << 6)
    out[:, 4:8] = (mn[:, 0:4] & 63) | ((mn[:, 4:8] >> 4) << 6)
    out[:, 8:12] = (sc[:, 4:8] & 15) | ((mn[:, 4:8] & 15) << 4)
    return out.astype(cp.uint8)


# ---------------------------------------------------------------- per-type value access
# Each codec: unpack(blocks)->vals (nb, QK) int32, pack(blocks, vals) in place,
# preds(blocks, x)->list of candidate predictions (nb, QK), bits.
def _nib32_unpack(qs):
    return cp.concatenate([qs & 15, qs >> 4], 1).astype(cp.int32)


def _nib32_pack(v):
    return (v[:, :16] | (v[:, 16:] << 4)).astype(cp.uint8)


def _q5_32_unpack(qh4, qs):
    qh = qh4.copy().view(cp.uint32).ravel()
    hb = ((qh[:, None] >> cp.arange(32, dtype=cp.uint32)[None, :]) & 1).astype(cp.int32)
    return _nib32_unpack(qs) | (hb << 4)


def _q5_32_pack(v):
    qh = ((v >> 4).astype(cp.uint32) << cp.arange(32, dtype=cp.uint32)[None, :]).sum(1, dtype=cp.uint32)
    return qh.view(cp.uint8).reshape(-1, 4), _nib32_pack(v & 15)


class Q8_0c:
    QK, BS, BITS = 32, 34, 8
    def sget(self, b): return {"d": (h_ord(get_h(b, 0)), 16)}
    def sset(self, b, f): set_h(b, 0, ord_h(f["d"]))
    def spred(self, x): return {"d": h_ord(f2h_bits(cp.abs(x).max(1) / cp.float32(127)))}
    def unpack(self, b): return b[:, 2:].astype(cp.int32)
    def pack(self, b, v): b[:, 2:] = (v & 255).astype(cp.uint8)
    def preds(self, b, x):
        d = h16(b[:, 0:2])
        amax = cp.abs(x).max(1); dr = amax / cp.float32(127)
        idr = safe_div(cp.float32(1), dr)
        return [roundf(x * idr[:, None]).astype(cp.int32) & 255,
                cp.clip(cp.rint(safe_div(x, d[:, None])), -127, 127).astype(cp.int32) & 255]


class Q4_0c:
    QK, BS, BITS = 32, 18, 4
    def sget(self, b): return {"d": (h_ord(get_h(b, 0)), 16)}
    def sset(self, b, f): set_h(b, 0, ord_h(f["d"]))
    def spred(self, x): return {"d": h_ord(f2h_bits(signed_max(x) / cp.float32(-8)))}
    def unpack(self, b): return _nib32_unpack(b[:, 2:18])
    def pack(self, b, v): b[:, 2:18] = _nib32_pack(v)
    def preds(self, b, x):
        d = h16(b[:, 0:2])
        am = cp.argmax(cp.abs(x), 1); mx = x[cp.arange(len(x)), am]
        dr = mx / cp.float32(-8); idr = safe_div(cp.float32(1), dr)
        p1 = cp.minimum(15, cp.floor(x * idr[:, None] + cp.float32(8.5))).astype(cp.int32)
        p2 = cp.clip(cp.rint(safe_div(x, d[:, None])) + 8, 0, 15).astype(cp.int32)
        return [cp.clip(p1, 0, 15), p2]


class Q4_1c:
    QK, BS, BITS = 32, 20, 4
    def sget(self, b): return {"d": (h_ord(get_h(b, 0)), 16), "m": (h_ord(get_h(b, 2)), 16)}
    def sset(self, b, f): set_h(b, 0, ord_h(f["d"])); set_h(b, 2, ord_h(f["m"]))
    def spred(self, x):
        mn = cp.minimum(x.min(1), 0)
        return {"d": h_ord(f2h_bits((x.max(1) - mn) / cp.float32(15))), "m": h_ord(f2h_bits(mn))}
    def unpack(self, b): return _nib32_unpack(b[:, 4:20])
    def pack(self, b, v): b[:, 4:20] = _nib32_pack(v)
    def preds(self, b, x):
        d, m = h16(b[:, 0:2]), h16(b[:, 2:4])
        mn, mxx = x.min(1), x.max(1); dr = (mxx - mn) / cp.float32(15); idr = safe_div(cp.float32(1), dr)
        p1 = cp.clip(cp.minimum(15, cp.floor((x - mn[:, None]) * idr[:, None] + cp.float32(0.5))), 0, 15).astype(cp.int32)
        p2 = cp.clip(cp.rint(safe_div(x - m[:, None], d[:, None])), 0, 15).astype(cp.int32)
        return [p1, p2]


class Q5_0c:
    QK, BS, BITS = 32, 22, 5
    def unpack(self, b): return _q5_32_unpack(b[:, 2:6], b[:, 6:22])
    def pack(self, b, v): b[:, 2:6], b[:, 6:22] = _q5_32_pack(v)
    def preds(self, b, x):
        d = h16(b[:, 0:2])
        return [cp.clip(cp.rint(safe_div(x, d[:, None])) + 16, 0, 31).astype(cp.int32)]


class Q5_1c:
    QK, BS, BITS = 32, 24, 5
    def unpack(self, b): return _q5_32_unpack(b[:, 4:8], b[:, 8:24])
    def pack(self, b, v): b[:, 4:8], b[:, 8:24] = _q5_32_pack(v)
    def preds(self, b, x):
        d, m = h16(b[:, 0:2]), h16(b[:, 2:4])
        return [cp.clip(cp.rint(safe_div(x - m[:, None], d[:, None])), 0, 31).astype(cp.int32)]


def _k2_index():
    """weight index -> (qs byte, shift) for the Q2_K / Q3_K low-bit layout"""
    byte = np.empty(256, np.int64); shift = np.empty(256, np.int64)
    for h in range(2):
        for j in range(4):
            for p in range(2):
                for l in range(16):
                    w = 128 * h + 32 * j + 16 * p + l
                    byte[w] = 32 * h + 16 * p + l; shift[w] = 2 * j
    return cp.asarray(byte), cp.asarray(shift)


K2_BYTE, K2_SHIFT = _k2_index()
K3_HBYTE = K2_BYTE % 32                                   # hmask byte: 16p + l
K3_HBIT = cp.asarray(np.array([4 * (w // 128) + (w % 128) // 32 for w in range(256)]))


def _pack_bits(v, byte_idx, shift, nbytes):
    # every (byte, shift) pair is unique, so OR-ing the fields is a sum
    flat = cp.zeros((v.shape[0], nbytes), cp.int64)
    for s in cp.unique(shift).get().tolist():
        sel = cp.nonzero(shift == s)[0]
        flat[:, byte_idx[sel]] += (v[:, sel].astype(cp.int64) << s)
    return flat.astype(cp.uint8)


class Q2_Kc:
    QK, BS, BITS = 256, 84, 2
    def sget(self, b):
        sc = b[:, 0:16].astype(cp.int32)
        return {"s": (sc & 15, 4), "m": (sc >> 4, 4), "d": (h_ord(get_h(b, 80)), 16), "dm": (h_ord(get_h(b, 82)), 16)}
    def sset(self, b, f):
        b[:, 0:16] = ((f["s"] & 15) | ((f["m"] & 15) << 4)).astype(cp.uint8)
        set_h(b, 80, ord_h(f["d"])); set_h(b, 82, ord_h(f["dm"]))
    def spred(self, x):
        sc, m = kmin_scales(x, 16, 3)
        ms, mm = sc.max(1), m.max(1)
        ls = cp.clip(cp.rint(safe_div(sc, ms[:, None] / 15)), 0, 15).astype(cp.int32)
        lm = cp.clip(cp.rint(safe_div(m, mm[:, None] / 15)), 0, 15).astype(cp.int32)
        return {"s": ls, "m": lm, "d": h_ord(f2h_bits(ms / 15)), "dm": h_ord(f2h_bits(mm / 15))}
    def unpack(self, b):
        qs = b[:, 16:80].astype(cp.int32)
        return (qs[:, K2_BYTE] >> K2_SHIFT[None, :]) & 3
    def pack(self, b, v): b[:, 16:80] = _pack_bits(v & 3, K2_BYTE, K2_SHIFT, 64)
    def preds(self, b, x):
        d, dmin = h16(b[:, 80:82]), h16(b[:, 82:84])
        sc = b[:, 0:16].astype(cp.int32)
        dl = (d[:, None] * (sc & 15).astype(cp.float32)); dm = (dmin[:, None] * (sc >> 4).astype(cp.float32))
        dl = cp.repeat(dl, 16, 1); dm = cp.repeat(dm, 16, 1)
        return [cp.where(dl != 0, cp.clip(cp.rint(safe_div(x + dm, dl)), 0, 3), 0).astype(cp.int32)]


class Q3_Kc:
    QK, BS, BITS = 256, 110, 3
    def _sc(self, b):
        s = b[:, 96:108].astype(cp.int32); j = np.arange(16)
        low = cp.where(cp.asarray(j < 8)[None, :], s[:, cp.asarray(np.where(j < 8, j, 0))] & 15,
                       s[:, cp.asarray(np.where(j < 8, 0, j - 8))] >> 4)
        high = (s[:, cp.asarray(8 + j % 4)] >> cp.asarray(2 * (j // 4))[None, :]) & 3
        return low | (high << 4)
    def sget(self, b): return {"s": (self._sc(b), 6), "d": (h_ord(get_h(b, 108)), 16)}
    def sset(self, b, f):
        v = f["s"] & 63
        out = cp.zeros((len(v), 12), cp.int32)
        out[:, 0:8] = (v[:, 0:8] & 15) | ((v[:, 8:16] & 15) << 4)
        for k in range(4):
            out[:, 8 + k] = sum(((v[:, 4 * q + k] >> 4) & 3) << (2 * q) for q in range(4))
        b[:, 96:108] = out.astype(cp.uint8)
        set_h(b, 108, ord_h(f["d"]))
    def spred(self, x):
        s = signed_max(x.reshape(len(x), 16, 16)) / cp.float32(-4)
        msc = signed_max(s)
        iscale = safe_div(cp.float32(-32), msc)
        l = cp.clip(cp.rint(iscale[:, None] * s), -32, 31).astype(cp.int32) + 32
        return {"s": l, "d": h_ord(f2h_bits(safe_div(cp.float32(1), iscale)))}
    def unpack(self, b):
        qs = b[:, 32:96].astype(cp.int32); hm = b[:, 0:32].astype(cp.int32)
        lo = (qs[:, K2_BYTE] >> K2_SHIFT[None, :]) & 3
        hi = (hm[:, K3_HBYTE] >> K3_HBIT[None, :]) & 1
        return lo | (hi << 2)
    def pack(self, b, v):
        b[:, 32:96] = _pack_bits(v & 3, K2_BYTE, K2_SHIFT, 64)
        b[:, 0:32] = _pack_bits((v >> 2) & 1, K3_HBYTE, K3_HBIT, 32)
    def preds(self, b, x):
        d = h16(b[:, 108:110]); s = b[:, 96:108].astype(cp.int32)
        j = np.arange(16)
        low = cp.where(cp.asarray(j < 8)[None, :], s[:, cp.asarray(np.where(j < 8, j, 0))] & 15,
                       s[:, cp.asarray(np.where(j < 8, 0, j - 8))] >> 4)
        high = (s[:, cp.asarray(8 + j % 4)] >> cp.asarray(2 * (j // 4))[None, :]) & 3
        sc = (low | (high << 4)) - 32
        dl = cp.repeat(d[:, None] * sc.astype(cp.float32), 16, 1)
        return [cp.where(dl != 0, cp.clip(cp.rint(safe_div(x, dl)), -4, 3) + 4, 4).astype(cp.int32)]


def _scale_min_k4(s):
    s = s.astype(cp.int32); nb = len(s)
    sc = cp.empty((nb, 8), cp.int32); mn = cp.empty_like(sc)
    sc[:, :4] = s[:, :4] & 63; mn[:, :4] = s[:, 4:8] & 63
    sc[:, 4:] = (s[:, 8:12] & 15) | ((s[:, 0:4] >> 6) << 4)
    mn[:, 4:] = (s[:, 8:12] >> 4) | ((s[:, 4:8] >> 6) << 4)
    return sc, mn


# Q4_K / Q5_K: weight 64c + 32*half + l <- qs[32c + l], low (half 0) / high (half 1) nibble
K4_BYTE = cp.asarray(np.array([32 * (w // 64) + w % 32 for w in range(256)]))
K4_SHIFT = cp.asarray(np.array([4 * ((w % 64) // 32) for w in range(256)]))
K5_HBYTE = cp.asarray(np.array([w % 32 for w in range(256)]))
K5_HBIT = cp.asarray(np.array([w // 32 for w in range(256)]))


def _kmin_preds(b, x, doff, bits):
    d, dmin = h16(b[:, doff:doff + 2]), h16(b[:, doff + 2:doff + 4])
    sc, mn = _scale_min_k4(b[:, doff + 4:doff + 16])
    dl = cp.repeat(d[:, None] * sc.astype(cp.float32), 32, 1)
    dm = cp.repeat(dmin[:, None] * mn.astype(cp.float32), 32, 1)
    return [cp.where(dl != 0, cp.clip(cp.rint(safe_div(x + dm, dl)), 0, (1 << bits) - 1), 0).astype(cp.int32)]


class KminScales:
    NMAX = 15
    def sget(self, b):
        sc, mn = _scale_min_k4(b[:, 4:16])
        return {"s": (sc, 6), "m": (mn, 6), "d": (h_ord(get_h(b, 0)), 16), "dm": (h_ord(get_h(b, 2)), 16)}
    def sset(self, b, f):
        b[:, 4:16] = pack_scale_min_k4(f["s"] & 63, f["m"] & 63)
        set_h(b, 0, ord_h(f["d"])); set_h(b, 2, ord_h(f["dm"]))
    def spred(self, x):
        sc, m = kmin_scales(x, 8, self.NMAX)
        ms, mm = sc.max(1), m.max(1)
        ls = cp.clip(cp.rint(safe_div(sc, ms[:, None] / 63)), 0, 63).astype(cp.int32)
        lm = cp.clip(cp.rint(safe_div(m, mm[:, None] / 63)), 0, 63).astype(cp.int32)
        return {"s": ls, "m": lm, "d": h_ord(f2h_bits(ms / 63)), "dm": h_ord(f2h_bits(mm / 63))}


class Q4_Kc(KminScales):
    QK, BS, BITS = 256, 144, 4
    def unpack(self, b): return (b[:, 16:144].astype(cp.int32)[:, K4_BYTE] >> K4_SHIFT[None, :]) & 15
    def pack(self, b, v): b[:, 16:144] = _pack_bits(v & 15, K4_BYTE, K4_SHIFT, 128)
    def preds(self, b, x): return _kmin_preds(b, x, 0, 4)


class Q5_Kc(KminScales):
    QK, BS, BITS = 256, 176, 5
    NMAX = 31
    def unpack(self, b):
        lo = (b[:, 48:176].astype(cp.int32)[:, K4_BYTE] >> K4_SHIFT[None, :]) & 15
        hi = (b[:, 16:48].astype(cp.int32)[:, K5_HBYTE] >> K5_HBIT[None, :]) & 1
        return lo | (hi << 4)
    def pack(self, b, v):
        b[:, 48:176] = _pack_bits(v & 15, K4_BYTE, K4_SHIFT, 128)
        b[:, 16:48] = _pack_bits((v >> 4) & 1, K5_HBYTE, K5_HBIT, 32)
    def preds(self, b, x): return _kmin_preds(b, x, 0, 5)


def _k6_index():
    lb = np.empty(256, np.int64); ls = np.empty(256, np.int64); hb = np.empty(256, np.int64); hs = np.empty(256, np.int64)
    for h in range(2):
        for k in range(4):
            for l in range(32):
                w = 128 * h + 32 * k + l
                lb[w] = 64 * h + l + (32 if k in (1, 3) else 0); ls[w] = 4 if k >= 2 else 0
                hb[w] = 32 * h + l; hs[w] = 2 * k
    return [cp.asarray(a) for a in (lb, ls, hb, hs)]


K6_LB, K6_LS, K6_HB, K6_HS = _k6_index()


class Q6_Kc:
    QK, BS, BITS = 256, 210, 6
    def sget(self, b): return {"s": (b[:, 192:208].astype(cp.int32), 8), "d": (h_ord(get_h(b, 208)), 16)}
    def sset(self, b, f): b[:, 192:208] = (f["s"] & 255).astype(cp.uint8); set_h(b, 208, ord_h(f["d"]))
    def spred(self, x):
        s = signed_max(x.reshape(len(x), 16, 16)) / cp.float32(-32)
        msc = signed_max(s)
        iscale = safe_div(cp.float32(-128), msc)
        l = cp.minimum(127, cp.rint(iscale[:, None] * s)).astype(cp.int32) & 255
        return {"s": l, "d": h_ord(f2h_bits(safe_div(cp.float32(1), iscale)))}
    def unpack(self, b):
        lo = (b[:, 0:128].astype(cp.int32)[:, K6_LB] >> K6_LS[None, :]) & 15
        hi = (b[:, 128:192].astype(cp.int32)[:, K6_HB] >> K6_HS[None, :]) & 3
        return lo | (hi << 4)
    def pack(self, b, v):
        b[:, 0:128] = _pack_bits(v & 15, K6_LB, K6_LS, 128)
        b[:, 128:192] = _pack_bits((v >> 4) & 3, K6_HB, K6_HS, 64)
    def preds(self, b, x):
        d = h16(b[:, 208:210]); sc = b[:, 192:208].view(cp.int8).astype(cp.float32)
        dl = cp.repeat(d[:, None] * sc, 16, 1)
        return [cp.where(dl != 0, cp.clip(cp.rint(safe_div(x, dl)), -32, 31) + 32, 32).astype(cp.int32)]


class IQ4_NLc:
    QK, BS, BITS = 32, 18, 4
    def sget(self, b): return {"d": (h_ord(get_h(b, 0)), 16)}
    def sset(self, b, f): set_h(b, 0, ord_h(f["d"]))
    def spred(self, x): return {"d": h_ord(f2h_bits(signed_max(x) / cp.float32(-127)))}
    def unpack(self, b): return _nib32_unpack(b[:, 2:18])
    def pack(self, b, v): b[:, 2:18] = _nib32_pack(v)
    def preds(self, b, x):
        d = h16(b[:, 0:2]); idl = safe_div(cp.float32(1), d)
        return [nearest_iq4(x * idl[:, None])]


IQ4X_BYTE = cp.asarray(np.array([16 * (w // 32) + w % 16 for w in range(256)]))
IQ4X_SHIFT = cp.asarray(np.array([4 * ((w % 32) // 16) for w in range(256)]))


class IQ4_XSc:
    QK, BS, BITS = 256, 136, 4
    def sget(self, b):
        sh = get_h(b, 2); sl = b[:, 4:8].astype(cp.int32); ib = np.arange(8)
        ls = ((sl[:, cp.asarray(ib // 2)] >> cp.asarray(4 * (ib % 2))[None, :]) & 15) | (((sh[:, None] >> cp.asarray(2 * ib)[None, :]) & 3) << 4)
        return {"s": (ls, 6), "d": (h_ord(get_h(b, 0)), 16)}
    def sset(self, b, f):
        ls = f["s"] & 63
        b[:, 4:8] = ((ls[:, 0::2] & 15) | ((ls[:, 1::2] & 15) << 4)).astype(cp.uint8)
        set_h(b, 2, sum(((ls[:, k] >> 4) & 3) << (2 * k) for k in range(8)))
        set_h(b, 0, ord_h(f["d"]))
    def spred(self, x):
        s = signed_max(x.reshape(len(x), 8, 32)) / cp.float32(-127)
        msc = signed_max(s)
        d = -msc / cp.float32(32)
        l = cp.clip(cp.rint(safe_div(s, d[:, None])), -32, 31).astype(cp.int32) + 32
        return {"s": l, "d": h_ord(f2h_bits(d))}
    def unpack(self, b): return (b[:, 8:136].astype(cp.int32)[:, IQ4X_BYTE] >> IQ4X_SHIFT[None, :]) & 15
    def pack(self, b, v): b[:, 8:136] = _pack_bits(v & 15, IQ4X_BYTE, IQ4X_SHIFT, 128)
    def preds(self, b, x):
        d = h16(b[:, 0:2])
        sh = b[:, 2:4].copy().view(cp.uint16).ravel().astype(cp.int32)
        sl = b[:, 4:8].astype(cp.int32)
        ib = np.arange(8)
        ls = ((sl[:, ib // 2] >> cp.asarray(4 * (ib % 2))[None, :]) & 15) | (((sh[:, None] >> cp.asarray(2 * ib)[None, :]) & 3) << 4)
        dl = d[:, None] * (ls - 32).astype(cp.float32)
        idl = cp.repeat(safe_div(cp.float32(1), dl), 32, 1)
        return [nearest_iq4(x * idl)]


SCALES = True            # predict scale fields too (method M27)

CODECS = {Q8_0: Q8_0c(), Q4_0: Q4_0c(), Q4_1: Q4_1c(), Q5_0: Q5_0c(), Q5_1: Q5_1c(), Q2_K: Q2_Kc(),
          Q3_K: Q3_Kc(), Q4_K: Q4_Kc(), Q5_K: Q5_Kc(), Q6_K: Q6_Kc(), IQ4_NL: IQ4_NLc(), IQ4_XS: IQ4_XSc()}


# ---------------------------------------------------------------- base weights
def base_float(base_dev, t):
    """BF16/F16/F32 tensor of the base file -> float32 device array"""
    raw = base_dev[t.offset:t.offset + t.nbytes]
    if t.type == BF16:
        return (raw.view(cp.uint16).astype(cp.uint32) << 16).view(cp.float32)
    if t.type == F16:
        return raw.view(cp.float16).astype(cp.float32)
    if t.type == F32:
        return raw.view(cp.float32)
    return None


def float_as(xf, ttype):
    """float32 -> raw bytes of a float type (round to nearest even)"""
    if ttype == F32:
        return xf.view(cp.uint8)
    if ttype == F16:
        return xf.astype(cp.float16).view(cp.uint8)
    if ttype == BF16:
        u = xf.view(cp.uint32)
        r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(cp.uint16)
        return r.view(cp.uint8)
    return None


def transform(qdev, qg, base_dev, base_by_name, inverse=False, stats=None, skip=()):
    """In-place on the device copy of a quantised file. Returns per-tensor plan for the archive."""
    plan = []
    for t in qg.tensors:
        if t.name in skip:
            continue
        bt = base_by_name.get(t.name)
        n = int(np.prod(t.shape))
        if bt is None or int(np.prod(bt.shape)) != n:
            continue
        x = base_float(base_dev, bt)
        if x is None:
            continue
        seg = qdev[t.offset:t.offset + t.nbytes]
        if inverse and isinstance(inverse.get(t.name), int) and inverse[t.name] >= 100:
            c = CODECS[t.type]
            b = seg.reshape(-1, c.BS)
            res, pred = c.sget(b), c.spred(x.reshape(-1, c.QK))
            c.sset(b, {n: (r + pred[n]) & ((1 << bits) - 1) for n, (r, bits) in res.items()})
        if t.type in (F32, F16, BF16):
            if inverse and inverse.get(t.name) != "xor":
                continue
            ref = float_as(x, t.type)
            if ref is not None and ref.size == seg.size:
                seg ^= ref
                plan.append([t.name, "xor"])
            continue
        c = CODECS.get(t.type)
        if c is None or t.nbytes != n // c.QK * c.BS:
            continue
        b = seg.reshape(-1, c.BS)
        xs = x.reshape(-1, c.QK)
        mask = (1 << c.BITS) - 1
        preds = c.preds(b, xs)
        if not inverse:
            v = c.unpack(b)
            errs = [int(((v - p) & mask).astype(bool).sum()) for p in preds]
            k = int(np.argmin(errs))
            c.pack(b, (v - preds[k]) & mask)
            if SCALES and hasattr(c, "spred"):
                act, pred = c.sget(b), c.spred(xs)
                c.sset(b, {n: (a - pred[n]) & ((1 << bits) - 1) for n, (a, bits) in act.items()})
                k += 100                                   # flag: scale fields are residuals
            plan.append([t.name, k])
            if stats is not None:
                s = stats.setdefault(t.type, [0, 0, 0])
                s[0] += v.size; s[1] += errs[k % 100]; s[2] += t.nbytes
        else:
            k = inverse.get(t.name)
            if k is None or not isinstance(k, int):
                continue
            c.pack(b, (c.unpack(b) + preds[k % 100]) & mask)
    return plan


# ---------------------------------------------------------------- archive
class M26Repo:
    name = "M26 whole model repo: BF16 + all quants (values predicted, scales stored)"

    def __init__(self, level=9):
        self.level = level
        self.base_codec = M24BF16GPU()
        self.stats = {}
        self.sizes = {}
        self.verbose = False
        self.dedup = True

    def compress(self, base: bytes, files: dict) -> bytes:
        """files: name -> bytes or path (read one at a time to bound memory)."""
        base_dev = cp.asarray(np.frombuffer(base, np.uint8))
        ch = {"base": pack(self.base_codec.compress_dev(base, base_dev))}
        self.sizes["BF16 (base)"] = len(ch["base"])
        bg = GGUF.from_bytes(base)
        by_name = {t.name: t for t in bg.tensors}
        zdict = zstandard.ZstdCompressionDict(base[:bg.data_start], dict_type=zstandard.DICT_TYPE_RAWCONTENT)
        params = zstandard.ZstdCompressionParameters.from_level(self.level, threads=4)
        manifest = []
        seen = {}                     # (tensor name, type, sha1) -> (file index, tensor name)
        for i, (name, src) in enumerate(files.items()):
            data = open(src, "rb").read() if isinstance(src, str) else src
            qg = GGUF.from_bytes(data)
            qdev = cp.asarray(np.frombuffer(data, np.uint8))
            dups = {}
            if self.dedup:
                for t in qg.tensors:
                    key = (t.name, t.type, hashlib.sha1(memoryview(data)[t.offset:t.offset + t.nbytes]).digest())
                    if key in seen:
                        dups[t.name] = seen[key]
                        qdev[t.offset:t.offset + t.nbytes] = 0
                    else:
                        seen[key] = (i, t.name)
            plan = transform(qdev, qg, base_dev, by_name, stats=self.stats, skip=dups)
            plan += [[tn, ["dup", fi, sn]] for tn, (fi, sn) in dups.items()]
            tdata = cp.asnumpy(qdev).tobytes()
            del qdev
            ch[f"f{i}"] = zstandard.ZstdCompressor(compression_params=params, dict_data=zdict).compress(tdata)
            self.sizes[name] = len(ch[f"f{i}"])
            if self.verbose:
                print(f"  {name}: {len(data):,} -> {self.sizes[name]:,} B", flush=True)
            manifest.append({"name": name, "size": len(data), "plan": plan})
            del data, tdata
            cp.get_default_memory_pool().free_all_blocks()
        ch["manifest"] = zstd_c(json.dumps(manifest).encode())
        return pack(ch)

    def decompress(self, buf: bytes, sink=None):
        """sink(name, uint8 array) is called per file; without it a dict is returned."""
        ch = unpack(buf)
        base_dev = self.base_codec.decompress_dev(unpack(ch["base"]))
        base = cp.asnumpy(base_dev)
        bg = GGUF.from_bytes(base[:64 << 20].tobytes())
        by_name = {t.name: t for t in bg.tensors}
        zdict = zstandard.ZstdCompressionDict(base[:bg.data_start].tobytes(), dict_type=zstandard.DICT_TYPE_RAWCONTENT)
        out = {}
        manifest = json.loads(zstd_d(ch["manifest"]))
        needed = {(k[1], k[2]) for m in manifest for _, k in m["plan"] if isinstance(k, list)}
        cache = {}
        for i, m in enumerate(manifest):
            tdata = zstandard.ZstdDecompressor(dict_data=zdict).decompress(ch[f"f{i}"], max_output_size=m["size"])
            qg = GGUF.from_bytes(tdata[:64 << 20])
            qdev = cp.asarray(np.frombuffer(tdata, np.uint8))
            del tdata
            plan = {name: k for name, k in m["plan"]}
            transform(qdev, qg, base_dev, by_name, inverse=plan)
            arr = cp.asnumpy(qdev)
            del qdev
            for t in qg.tensors:
                k = plan.get(t.name)
                if isinstance(k, list):
                    arr[t.offset:t.offset + t.nbytes] = cache[(k[1], k[2])]
                if (i, t.name) in needed:
                    cache[(i, t.name)] = arr[t.offset:t.offset + t.nbytes].copy()
            cp.get_default_memory_pool().free_all_blocks()
            if sink:
                sink(m["name"], arr)
            else:
                out[m["name"]] = arr
        return base, out
