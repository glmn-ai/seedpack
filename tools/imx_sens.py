import os, sys
sys.path.insert(0, ".")
import numpy as np, cupy as cp
from seedpack.gguf_reader import GGUF
from seedpack.requant import QFUN, load_imatrix
from seedpack.gpuquant import GPUQuant
d = "models/qwen35-2b"
base = open(os.path.join(d, "Qwen3.5-2B-BF16.gguf"), "rb").read()
by = {t.name: t for t in GGUF.from_bytes(base, light=True).tensors}
imx = load_imatrix(os.path.join(d, "imatrix_unsloth.gguf_file"))
g = GPUQuant(); rng = np.random.default_rng(0)
names = ["blk.0.ffn_down.weight", "blk.3.ffn_up.weight", "blk.0.ssm_out.weight", "blk.5.attn_gate.weight"]
for ty in (2, 14, 13):
    xs = [(cp.asarray(np.frombuffer(base, np.uint8, by[n].nbytes, by[n].offset)).view(cp.uint16).astype(cp.uint32) << 16).view(cp.float32).reshape(-1, by[n].shape[0]) for n in names]
    ref = [cp.asnumpy(a) for a in g.quantize_batch(ty, xs, [imx[n] for n in names])]
    bs = QFUN[ty][2]
    for lab, f in (("1 ulp", lambda v: np.nextafter(v, np.where(rng.random(v.size) < .5, np.inf, -np.inf)).astype(np.float32)),
                   ("1e-4", lambda v: (v * (1 + 1e-4 * rng.standard_normal(v.size))).astype(np.float32)),
                   ("1e-2", lambda v: (v * (1 + 1e-2 * rng.standard_normal(v.size))).astype(np.float32)),
                   ("x1.5", lambda v: (v * np.float32(1.5)).astype(np.float32))):
        out = g.quantize_batch(ty, xs, [f(imx[n]) for n in names])
        r = np.mean([(cp.asnumpy(o).reshape(-1, bs) != rr.reshape(-1, bs)).any(1).mean() for o, rr in zip(out, ref)])
        print(f"type {ty:2d} imatrix noise {lab:6s}: {100 * r:.2f}% blocks differ")
