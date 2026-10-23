import os, sys, time
sys.path.insert(0, ".")
import numpy as np, cupy as cp
from seedpack.gguf_reader import GGUF
from seedpack.requant import load_imatrix
from seedpack.gpuquant import GPUQuant
d = "models/unsloth"
base = open(os.path.join(d, "Qwen3-0.6B-BF16.gguf"), "rb").read()
bg = GGUF.from_bytes(base); by = {t.name: t for t in bg.tensors}
bd = cp.asarray(np.frombuffer(base, np.uint8))
imx = load_imatrix(os.path.join(d, "imatrix_unsloth.dat"))
raw = open(os.path.join(d, "Qwen3-0.6B-UD-IQ1_M.gguf"), "rb").read(); qg = GGUF.from_bytes(raw[:64 << 20])
g = GPUQuant()
ts = [t for t in qg.tensors if t.type == 29]
xs = [(bd[by[t.name].offset:by[t.name].offset + by[t.name].nbytes].view(cp.uint16).astype(cp.uint32) << 16).view(cp.float32).reshape(-1, t.shape[0]) for t in ts]
for v in (0, 1):
    cp.cuda.Device().synchronize(); t0 = time.perf_counter()
    out = [cp.asnumpy(o) for o in g.quantize_batch(29, xs, [imx.get(t.name) for t in ts], variant=v)]
    dt = time.perf_counter() - t0
    bad = sum(int((o != np.frombuffer(raw, np.uint8, t.nbytes, t.offset)).any()) for o, t in zip(out, ts))
    print(f"variant {v}: {len(ts)} IQ1_M tensors, not exact {bad}, {dt:.2f} s")
