import os, sys
sys.path.insert(0, ".")
import numpy as np, cupy as cp
from seedpack.gguf_reader import GGUF
from seedpack.requant import QFUN
from seedpack.gpuquant import GPUQuant
d = "models/qwen35-2b"
base = open(os.path.join(d, "Qwen3.5-2B-BF16.gguf"), "rb").read()
bg = GGUF.from_bytes(base, light=True); by = {t.name: t for t in bg.tensors}
bd = None
ig = GGUF(os.path.join(d, "imatrix_unsloth.gguf_file"), light=True); ib = ig.data
print({k: v for k, v in ig.meta.items() if v is not None and not isinstance(v, list)}, [k for k, v in ig.meta.items() if isinstance(v, list) or v is None])
T = {t.name: t for t in ig.tensors}
q = open(os.path.join(d, "Qwen3.5-2B-Q4_K_M.gguf"), "rb").read(); qg = GGUF.from_bytes(q, light=True)
g = GPUQuant()
name = "blk.0.attn_gate.weight"
t = next(x for x in qg.tensors if x.name == name)
sums = np.frombuffer(ib, np.float32, T[name + ".in_sum2"].nbytes // 4, T[name + ".in_sum2"].offset)
cnt = np.frombuffer(ib, np.float32, 1, T[name + ".counts"].offset)[0]
bt = by[name]
x = (cp.asarray(np.frombuffer(base, np.uint8, bt.nbytes, bt.offset)).view(cp.uint16).astype(cp.uint32) << 16).view(cp.float32).reshape(-1, t.shape[0])
ref = np.frombuffer(q, np.uint8, t.nbytes, t.offset).reshape(-1, 144)
def f(im):
    p = cp.asnumpy(g.quantize_batch(t.type, [x], [np.asarray(im, np.float32)])[0]).reshape(-1, 144)
    return (p != ref).any(1).mean()
c32 = np.float32(cnt)
alts = {"s/c": sums / c32, "s*(1/c)": sums * (np.float32(1) / c32), "double": (sums.astype(np.float64) / float(cnt)).astype(np.float32)}
for nc in (80, 81, 79):
    alts[f"legacy ncall={nc}"] = ((sums / c32) * np.float32(nc)) / np.float32(nc)
for k, v in alts.items():
    print(f"{k:20s} {f(v):.4f}")
