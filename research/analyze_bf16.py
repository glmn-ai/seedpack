"""Entropy study of BF16 weights: ideal code length (empirical conditional entropy, per-tensor
static tables) of the M24 context model vs richer contexts. CPU/numpy only.

usage: analyze_bf16.py [gguf] [max tensors]
"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from seedpack.gguf_reader import GGUF, GGML_BF16

path = sys.argv[1] if len(sys.argv) > 1 else "models/unsloth/Qwen3-0.6B-BF16.gguf"
maxt = int(sys.argv[2]) if len(sys.argv) > 2 else 10**9
g = GGUF(path)
ts = sorted((t for t in g.tensors if t.type == GGML_BF16 and len(t.shape) == 2), key=lambda t: t.offset)[:maxt]


def H(sym, ctx, nsym):
    """sum over contexts of n * H(sym | ctx) in bits, + number of non-zero table cells"""
    k = ctx.astype(np.int64).ravel() * nsym + sym.astype(np.int64).ravel()
    c = np.bincount(k, minlength=int(k.max()) + 1).astype(np.float64)
    c = c[: (len(c) + nsym - 1) // nsym * nsym] if len(c) % nsym == 0 else np.pad(c, (0, nsym - len(c) % nsym))
    c = c.reshape(-1, nsym)
    tot = c.sum(1, keepdims=True)
    nz = c > 0
    bits = -(c[nz] * np.log2((c / np.maximum(tot, 1))[nz])).sum()
    return bits, int(nz.sum())


acc = {}


def add(name, bits_cells):
    b, c = bits_cells
    a = acc.setdefault(name, [0.0, 0])
    a[0] += b; a[1] += c


N = 0
t0 = time.time()
for t in ts:
    rowlen = t.shape[0]
    u = np.frombuffer(g.data, np.uint16, t.nbytes // 2, t.offset).reshape(-1, rowlen)
    N += u.size
    w = (u.astype(np.uint32) << 16).view(np.float32)
    rf = np.log2(np.sqrt((w.astype(np.float64) ** 2).mean(1)) + 1e-30)
    wn = w / np.exp2(rf)[:, None]
    cf = np.log2(np.sqrt((wn.astype(np.float64) ** 2).mean(0)) + 1e-30)
    rfi = np.rint(rf * 4).astype(np.int32); cfi = np.rint(cf * 4).astype(np.int32)
    bias = wn.mean(0) / np.exp2(cfi / 4.0)
    sc = np.clip(np.floor(bias * 16) + 8, 0, 15).astype(np.int32)
    ex = ((u >> 7) & 255).astype(np.int32); man = (u & 127).astype(np.int32); sg = (u >> 15).astype(np.int32)
    pred4 = rfi[:, None] + cfi[None, :]
    eb = np.clip((pred4 + 108) >> 1, 0, 127)
    rel = np.clip(ex - ((pred4 + 508) >> 2) + 8, 0, 15)
    # M24 model
    add("E0 exp|eb (M24)", H(ex, eb, 256))
    add("M0 man|rel (M24)", H(man, rel, 128))
    add("S0 sgn|scl (M24)", H(sg, np.broadcast_to(sc[None, :], u.shape), 2))
    # --- exponent alternatives
    pq = np.clip(pred4 + 216, 0, 255)                         # quarter-octave predictor
    add("E1 exp|pred4 (1/4 oct)", H(ex, pq, 256))
    # relative exponent of left neighbour (same row) and of the row above (same column)
    r4 = np.clip(4 * ex - (pred4 + 508) + 16, 0, 47) // 4      # coarse rel in octaves 0..11
    left = np.zeros_like(r4); left[:, 1:] = r4[:, :-1]
    up = np.zeros_like(r4); up[1:, :] = r4[:-1, :]
    add("E2 exp|eb,left", H(ex, eb * 12 + left, 256))
    add("E3 exp|eb,up", H(ex, eb * 12 + up, 256))
    add("E4 exp|pred4,left,up", H(ex, (pq * 12 + left) * 12 + up, 256))
    # --- mantissa alternatives
    rel4 = np.clip(4 * ex - (pred4 + 508) + 32, 0, 63)           # quarter-octave rel
    add("M1 man|rel4", H(man, rel4, 128))
    add("M2 man|rel4,ex==0?", H(man, rel4 * 2 + (ex == 0), 128))
    hi = man >> 4
    add("M3a man_hi3|rel4", H(hi, rel4, 8))
    add("M3b man_lo4|rel4,hi", H(man & 15, rel4 * 8 + hi, 16))
    lm = np.zeros_like(man); lm[:, 1:] = man[:, :-1] >> 5
    add("M4 man|rel4,left_man_hi2", H(man, rel4 * 4 + lm, 128))
    # --- sign alternatives
    ls = np.zeros_like(sg); ls[:, 1:] = sg[:, :-1]
    us = np.zeros_like(sg); us[1:, :] = sg[:-1, :]
    add("S1 sgn|scl,left", H(sg, sc[None, :] * 2 + ls, 2))
    add("S2 sgn|scl,up", H(sg, sc[None, :] * 2 + us, 2))
    add("S3 sgn|col (per-column table)", H(sg, np.broadcast_to(np.arange(rowlen)[None, :], u.shape), 2))
    add("S4 sgn|row", H(sg, np.broadcast_to(np.arange(u.shape[0])[:, None], u.shape), 2))
    print(f"{t.name:32s} {u.shape} {time.time() - t0:.0f}s", flush=True)

print(f"\n{N:,} weights; bits/weight (ideal, excl. tables) and table cells")
for k, (b, c) in acc.items():
    print(f"{k:34s} {b / N:8.4f} bpw  {b / 8 / 1e6:10.2f} MB  cells {c:,}")
