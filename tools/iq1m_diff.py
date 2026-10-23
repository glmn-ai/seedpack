import os, sys
sys.path.insert(0, "src")
import numpy as np, cupy as cp
from gguf_reader import GGUF
from requant import quantize, load_imatrix
from repo2 import base_float_np
from gpuquant import GPUQuant
d = "models/unsloth"
base = open(os.path.join(d, "Qwen3-0.6B-BF16.gguf"), "rb").read()
bg = GGUF.from_bytes(base); by = {t.name: t for t in bg.tensors}
imx = load_imatrix(os.path.join(d, "imatrix_unsloth.dat"))
p = os.path.join(d, "Qwen3-0.6B-UD-IQ1_M.gguf")
raw = open(p, "rb").read(); qg = GGUF.from_bytes(raw[:64 << 20])
g = GPUQuant()
tot = [0, 0, 0]
for t in [t for t in qg.tensors if t.type == 29][:12]:
    x = base_float_np(base, by[t.name]).reshape(-1, t.shape[0])
    ref = np.frombuffer(raw, np.uint8, t.nbytes, t.offset).reshape(-1, 56)
    c = quantize(29, x, imx.get(t.name)).reshape(-1, 56)
    gg = cp.asnumpy(g.quantize_batch(29, [cp.asarray(x)], [imx.get(t.name)])[0]).reshape(-1, 56)
    r = [(c != ref).any(1).sum(), (gg != ref).any(1).sum(), (c != gg).any(1).sum()]
    print(f"{t.name:28s} blocks {len(ref):6d}  cpu!=file {r[0]:4d}  gpu!=file {r[1]:4d}  cpu!=gpu {r[2]:4d}")
