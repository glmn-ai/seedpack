import os, sys
sys.path.insert(0, ".")
import numpy as np, cupy as cp
from seedpack.gguf_reader import GGUF
from seedpack.requant import QFUN
from seedpack.gpuquant import GPUQuant
d = "models/qwen35-2b"
base = open(os.path.join(d, "Qwen3.5-2B-BF16.gguf"), "rb").read()
by = {t.name: t for t in GGUF.from_bytes(base, light=True).tensors}
ig = GGUF(os.path.join(d, "imatrix_unsloth.gguf_file"), light=True); ib = ig.data
T = {t.name: t for t in ig.tensors}
q = open(os.path.join(d, "Qwen3.5-2B-Q6_K.gguf"), "rb").read(); qt = {t.name: t for t in GGUF.from_bytes(q, light=True).tensors}
g = GPUQuant()
name = "blk.0.ffn_down.weight"; t = qt[name]; bt = by[name]
sums = np.frombuffer(ib, np.float32, T[name + ".in_sum2"].nbytes // 4, T[name + ".in_sum2"].offset)
cnt = float(np.frombuffer(ib, np.float32, 1, T[name + ".counts"].offset)[0])
print("count", cnt, "all counts:", sorted({float(np.frombuffer(ib, np.float32, 1, x.offset)[0]) for x in ig.tensors if x.name.endswith(".counts")})[:6])
x = (cp.asarray(np.frombuffer(base, np.uint8, bt.nbytes, bt.offset)).view(cp.uint16).astype(cp.uint32) << 16).view(cp.float32).reshape(-1, t.shape[0])
ref = np.frombuffer(q, np.uint8, t.nbytes, t.offset).reshape(-1, 210)
def f(im):
    return 100 * (cp.asnumpy(g.quantize_batch(14, [x], [np.asarray(im, np.float32)])[0]).reshape(-1, 210) != ref).any(1).mean()
cands = {"published s/c": sums / np.float32(cnt), "s (no division)": sums, "s/901120": sums / np.float32(901120),
         "s/80": sums / np.float32(80), "s/c*80": sums / np.float32(cnt) * np.float32(80), "s/c*11264": sums / np.float32(cnt) * np.float32(11264),
         "s/c/80": sums / np.float32(cnt) / np.float32(80), "s/(c/11264)": sums / np.float32(cnt / 11264)}
for k, v in cands.items():
    print(f"{k:18s} {f(v):6.2f}% blocks differ")
