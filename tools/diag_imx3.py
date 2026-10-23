import os, sys
sys.path.insert(0, ".")
import numpy as np, cupy as cp
from seedpack.gguf_reader import GGUF
from seedpack.requant import load_imatrix
from seedpack.gpuquant import GPUQuant
d = "models/qwen35-2b"
base = open(os.path.join(d, "Qwen3.5-2B-BF16.gguf"), "rb").read()
by = {t.name: t for t in GGUF.from_bytes(base, light=True).tensors}
imx = load_imatrix(os.path.join(d, "imatrix_unsloth.gguf_file"))
q = open(os.path.join(d, "Qwen3.5-2B-Q4_K_M.gguf"), "rb").read(); qt = {t.name: t for t in GGUF.from_bytes(q, light=True).tensors}
g = GPUQuant(fmad=len(sys.argv) > 1)
for name, bs in (("blk.0.ffn_down.weight", 210), ("blk.0.attn_gate.weight", 144), ("blk.0.ssm_out.weight", 176)):
    t = qt[name]; bt = by[name]
    x = (cp.asarray(np.frombuffer(base, np.uint8, bt.nbytes, bt.offset)).view(cp.uint16).astype(cp.uint32) << 16).view(cp.float32).reshape(-1, t.shape[0])
    p = cp.asnumpy(g.quantize_batch(t.type, [x], [imx[name]])[0])
    nb = t.shape[0] // 256
    diff = (p.reshape(-1, bs) != np.frombuffer(q, np.uint8, t.nbytes, t.offset).reshape(-1, bs)).any(1).reshape(-1, nb)
    col = diff.mean(0)
    print(name, "rows", diff.shape[0], "diff rate per superblock column:", np.round(col, 3))
    print("   rows with any diff:", diff.any(1).mean().round(3), " correlation across rows (if imatrix differs, columns stand out)")
