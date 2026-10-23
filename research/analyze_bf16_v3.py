"""Does anything beyond the rank-1 variance model help? (on a subset of tensors, K=8 contexts)
  - oracle tile scales (side info cost NOT included): residual log-rms per tile added to the prediction
  - sign | (column bias class, relative magnitude)
  - exponent | (pe, sign of column bias) ...
usage: analyze_bf16_v3.py [gguf] [tensor name regex]
"""
import sys, os, re
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from gguf_reader import GGUF, GGML_BF16

path = sys.argv[1] if len(sys.argv) > 1 else "models/unsloth/Qwen3-0.6B-BF16.gguf"
rx = re.compile(sys.argv[2] if len(sys.argv) > 2 else r"blk\.(0|9|20)\.")
g = GGUF(path, light=True) if "light" in GGUF.__init__.__code__.co_varnames else GGUF(path)
ts = [t for t in g.tensors if t.type == GGML_BF16 and len(t.shape) == 2 and rx.search(t.name)]


def H(sym, ctx, nsym):
    k = ctx.astype(np.int64).ravel() * nsym + sym.astype(np.int64).ravel()
    c = np.bincount(k).astype(np.float64)
    c = np.pad(c, (0, (-len(c)) % nsym)).reshape(-1, nsym)
    tot = c.sum(1, keepdims=True); nz = c > 0
    return -(c[nz] * np.log2((c / np.maximum(tot, 1))[nz])).sum()


acc = {}
N = 0
for t in ts:
    rl = t.shape[0]
    u = np.frombuffer(g.data, np.uint16, t.nbytes // 2, t.offset).reshape(-1, rl)
    N += u.size
    w = (u.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    rf = np.log2(np.sqrt((w ** 2).mean(1)) + 1e-30)
    wn = w / np.exp2(rf)[:, None]
    cf = np.log2(np.sqrt((wn ** 2).mean(0)) + 1e-30)
    rfi = np.rint(rf * 8).astype(np.int64); cfi = np.rint(cf * 8).astype(np.int64)
    pred = rfi[:, None] + cfi[None, :]
    bias = wn.mean(0) / np.exp2(cfi / 8)
    sc = np.clip(np.floor(bias * 16) + 8, 0, 15).astype(np.int64)
    ex = ((u >> 7) & 255).astype(np.int64); man = (u & 127).astype(np.int64); sg = (u >> 15).astype(np.int64)

    def add(k, v):
        acc[k] = acc.get(k, 0) + v

    pe = np.clip(pred + 256, 0, 511)
    rlc = np.clip(8 * ex - pred - 1016 + 64, 0, 127)
    add("E  exp|pe", H(ex, pe, 256))
    add("M  man|rl", H(man, rlc, 128))
    add("S  sgn|scl", H(sg, np.broadcast_to(sc[None, :], u.shape), 2))
    rb = np.clip(rlc // 16, 0, 7)
    add("S2 sgn|scl,rl/2oct", H(sg, sc[None, :] * 8 + rb, 2))
    # finer column bias: 64 classes
    sc64 = np.clip(np.floor(bias * 64) + 32, 0, 63).astype(np.int64)
    add("S3 sgn|scl64,rl/2oct", H(sg, sc64[None, :] * 8 + rb, 2))
    # oracle tile scales
    z = wn / np.exp2(cfi / 8)[None, :]
    for th, tw in ((1, 128), (32, 32), (16, 256), (256, 1)):
        nr, nc = u.shape[0] // th * th, rl // tw * tw
        zz = z[:nr, :nc].reshape(nr // th, th, nc // tw, tw)
        tr = np.rint(8 * np.log2(np.sqrt((zz ** 2).mean((1, 3))) + 1e-30)).astype(np.int64)
        p2 = pred[:nr, :nc] + np.repeat(np.repeat(tr, th, 0), tw, 1)
        add(f"E+ oracle tile {th}x{tw}", H(ex[:nr, :nc], np.clip(p2 + 256, 0, 511), 256) + H(ex[nr:], pe[nr:], 256) * 0)
        add(f"E  same region {th}x{tw}", H(ex[:nr, :nc], pe[:nr, :nc], 256))
        add(f"   tiles {th}x{tw} count", tr.size * 0 + (nr // th) * (nc // tw))
    print(t.name, flush=True)

print(f"{N:,} weights")
for k, v in acc.items():
    print(f"{k:28s} {v / N:9.4f} bpw" if "count" not in k else f"{k:28s} {v:,} tiles")
