"""Which tensors of a Qwen3.5 quant differ from the GPU re-quantisation, and by how much; tries imatrix normalisations."""
import os, sys
sys.path.insert(0, "src")
import numpy as np, cupy as cp
from gguf_reader import GGUF, TYPE_NAMES
from requant import QFUN, load_imatrix
from gpuquant import GPUQuant
from gguf_reader import GGUF as G
d = "models/qwen35-2b"
base = open(os.path.join(d, "Qwen3.5-2B-BF16.gguf"), "rb").read()
bg = GGUF.from_bytes(base, light=True); by = {t.name: t for t in bg.tensors}
bd = cp.asarray(np.frombuffer(base, np.uint8))
imx = load_imatrix(os.path.join(d, "imatrix_unsloth.gguf_file"))
ig = G(os.path.join(d, "imatrix_unsloth.gguf_file")); ib = ig.data
raw_sums = {t.name[:-8]: np.frombuffer(ib, np.float32, t.nbytes // 4, t.offset) for t in ig.tensors if t.name.endswith(".in_sum2")}
counts = {t.name[:-7]: np.frombuffer(ib, np.float32, t.nbytes // 4, t.offset) for t in ig.tensors if t.name.endswith(".counts")}
q = open(os.path.join(d, "Qwen3.5-2B-Q4_K_M.gguf"), "rb").read(); qg = GGUF.from_bytes(q, light=True)
g = GPUQuant()
def frac(t, im):
    bt = by[t.name]
    x = (bd[bt.offset:bt.offset + bt.nbytes].view(cp.uint16).astype(cp.uint32) << 16).view(cp.float32).reshape(-1, t.shape[0])
    p = cp.asnumpy(g.quantize_batch(t.type, [x], [im])[0])
    bs = QFUN[t.type][2]
    ref = np.frombuffer(q, np.uint8, t.nbytes, t.offset)
    return (p.reshape(-1, bs) != ref.reshape(-1, bs)).any(1).mean()
for t in [t for t in qg.tensors if t.type in QFUN][:12]:
    n = t.name
    im = imx.get(n)
    c = counts.get(n)
    alts = {"cur": im, "none": None}
    if n in raw_sums:
        alts["raw"] = raw_sums[n].copy()
        alts["x chunk"] = (raw_sums[n] / c[0] * 11264).astype(np.float32) if c is not None else None
    res = {k: f"{frac(t, v):.4f}" for k, v in alts.items() if not (k != "none" and v is None)}
    print(f"{n:32s} {TYPE_NAMES[t.type]:6s} shape={t.shape} count={None if c is None else c[:3]} diff-blocks: {res}")
