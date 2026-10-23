"""Numpy unpackers of quant blocks: codes, affine (value = a * code + c) and scale-field bytes."""
import numpy as np


def f16(b, o):
    return b[:, o:o + 2].copy().view(np.float16)[:, 0].astype(np.float32)


# ---------------------------------------------------------------- per type: (scale bytes, codes, step, offset)
def q4_0(b):
    """-> codes (nb, 32), affine (a, c) so that value = a * code + c per weight, scale-field bytes"""
    d = f16(b, 0)
    qs = b[:, 2:18]
    q = np.concatenate([qs & 15, qs >> 4], 1).astype(np.int32)
    a = np.repeat(d[:, None], 32, 1)
    return q, a, -8 * a, b[:, 0:2]


def scale_min_k4(s):
    sc = np.zeros((len(s), 8), np.float32); m = np.zeros((len(s), 8), np.float32)
    for j in range(8):
        if j < 4:
            sc[:, j] = s[:, j] & 63; m[:, j] = s[:, j + 4] & 63
        else:
            sc[:, j] = (s[:, j + 4] & 0xF) | ((s[:, j - 4] >> 6) << 4)
            m[:, j] = (s[:, j + 4] >> 4) | ((s[:, j] >> 6) << 4)
    return sc, m


def q4_k(b):
    d, dmin = f16(b, 0), f16(b, 2)
    sc, m = scale_min_k4(b[:, 4:16])
    qs = b[:, 16:144].reshape(-1, 4, 32)
    q = np.stack([qs & 15, qs >> 4], 2).reshape(-1, 256).astype(np.int32)      # sub-block j = 2c + (0 low, 1 high)
    a = np.repeat(d[:, None] * sc, 32, 1)
    c = -np.repeat(dmin[:, None] * m, 32, 1)
    return q, a, c, b[:, 0:16]


def q6_k(b):
    ql, qh, s = b[:, 0:128], b[:, 128:192], b[:, 192:208].view(np.int8).astype(np.float32)
    d = f16(b, 208)
    q = np.zeros((len(b), 256), np.int32); a = np.zeros((len(b), 256), np.float32)
    for n in range(2):
        for g in range(4):
            l = np.arange(32)
            lo = ql[:, 64 * n + l + (32 if g in (1, 3) else 0)]
            nib = (lo & 15) if g in (0, 1) else (lo >> 4)
            hi = (qh[:, 32 * n + l] >> (2 * g)) & 3
            w = 128 * n + 32 * g + l
            q[:, w] = nib | (hi << 4)
            a[:, w] = d[:, None] * s[:, 8 * n + l // 16 + 2 * g]
    return q, a, -32 * a, b[:, 192:210]
