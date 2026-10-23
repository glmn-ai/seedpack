"""Fraction of differing quant blocks per tensor type between a published GGUF and the GPU re-quantisation."""
import os, sys, collections
sys.path.insert(0, "src")
import numpy as np, cupy as cp
from gguf_reader import GGUF, TYPE_NAMES
from requant import QFUN, load_imatrix
from gpuquant import GPUQuant
d, files = sys.argv[1], sys.argv[2:]
base_p = next(os.path.join(d, f) for f in os.listdir(d) if f.endswith("-BF16.gguf") and not f.startswith("mmproj"))
base = open(base_p, "rb").read()
by = {t.name: t for t in GGUF.from_bytes(base, light=True).tensors}
imx = load_imatrix(next(os.path.join(d, f) for f in os.listdir(d) if f.startswith("imatrix")))
g = GPUQuant()
for fn in files:
    q = open(os.path.join(d, fn), "rb").read()
    st = collections.defaultdict(lambda: [0, 0, 0, 0])
    worst = []
    for t in GGUF.from_bytes(q, light=True).tensors:
        if t.type not in QFUN or t.name not in by:
            continue
        bt = by[t.name]
        x = (cp.asarray(np.frombuffer(base, np.uint8, bt.nbytes, bt.offset)).view(cp.uint16).astype(cp.uint32) << 16).view(cp.float32).reshape(-1, t.shape[0])
        im = imx.get(t.name)
        p = cp.asnumpy(g.quantize_batch(t.type, [x], [im])[0])
        bs = QFUN[t.type][2]
        dif = (p.reshape(-1, bs) != np.frombuffer(q, np.uint8, t.nbytes, t.offset).reshape(-1, bs)).any(1)
        s = st[TYPE_NAMES[t.type]]; s[0] += 1; s[1] += dif.any(); s[2] += dif.sum(); s[3] += dif.size
        worst.append((dif.mean(), t.name, TYPE_NAMES[t.type], im is not None))
    print(fn, {k: f"{v[1]}/{v[0]} tensors, {100 * v[2] / v[3]:.2f}% blocks" for k, v in st.items()})
    print("   worst:", [(f"{a:.3f}", n, ty, imok) for a, n, ty, imok in sorted(worst, reverse=True)[:4]])
