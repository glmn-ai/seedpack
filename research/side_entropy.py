"""How small can R2's side info be? zstd vs order-0 / delta entropy of rfi, cfi, scl."""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, zstandard
from seedpack.codecs_gguf import unpack, zstd_d
from seedpack.bf16v2 import BF16v2
data = open("models/unsloth/Qwen3-0.6B-BF16.gguf", "rb").read()
arc = BF16v2().compress(data); ch = unpack(arc)
meta = json.loads(zstd_d(ch["meta"])); lay = meta["layout"]
_, _, nrow, ncol = BF16v2._bases(lay)
side = zstd_d(ch["side"])
rfi = np.frombuffer(side, np.int16, nrow, 0).astype(np.int64); cfi = np.frombuffer(side, np.int16, ncol, 2 * nrow).astype(np.int64)
scl = np.frombuffer(side, np.int8, ncol, 2 * nrow + 2 * ncol).astype(np.int64)
def ent(x):
    _, c = np.unique(x, return_counts=True); p = c / c.sum(); return -(c * np.log2(p)).sum() / 8
print("zstd side", len(ch["side"]))
# per tensor: subtract tensor median
rb, cb, _, _ = BF16v2._bases(lay)
rr, cc = [], []
for i, (off, nb, rl, _) in enumerate(lay["t"]):
    r = rfi[rb[i]:rb[i] + nb // 2 // rl]; c = cfi[cb[i]:cb[i] + rl]
    rr.append(r - np.median(r).astype(np.int64)); cc.append(c - np.median(c).astype(np.int64))
rr = np.concatenate(rr); cc = np.concatenate(cc)
print("order0 rfi-med", ent(rr), "cfi-med", ent(cc), "scl", ent(scl), "total", ent(rr) + ent(cc) + ent(scl))
d = np.diff(rr, prepend=0); print("rfi delta", ent(d))
print("rfi raw zstd", len(zstandard.ZstdCompressor(level=19).compress(rfi.astype(np.int16).tobytes())))
