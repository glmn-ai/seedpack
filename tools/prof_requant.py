"""Where does M27 spend its time? Re-quantisation per ggml type over all unique (tensor, type) pairs."""
import glob, os, sys, time, collections
sys.path.insert(0, ".")
import numpy as np
from seedpack.gguf_reader import GGUF, TYPE_NAMES
from seedpack.requant import QFUN, quantize, load_imatrix
from seedpack.repo2 import base_float_np

d = "models/unsloth"
base = open(os.path.join(d, "Qwen3-0.6B-BF16.gguf"), "rb").read()
bg = GGUF.from_bytes(base); by = {t.name: t for t in bg.tensors}
imx = load_imatrix(os.path.join(d, "imatrix_unsloth.dat"))
pairs = set()
for p in sorted(glob.glob(os.path.join(d, "*.gguf"))):
    with open(p, "rb") as f:
        for t in GGUF.from_bytes(f.read(64 << 20)).tensors:
            if t.type in QFUN:
                pairs.add((t.name, t.type, t.shape[0]))
print("unique (tensor,type):", len(pairs))
tm, cnt, el = collections.Counter(), collections.Counter(), collections.Counter()
T0 = time.perf_counter()
for name, ty, n0 in sorted(pairs):
    x = base_float_np(base, by[name])
    t0 = time.perf_counter(); quantize(ty, x.reshape(-1, n0), imx.get(name)); dt = time.perf_counter() - t0
    k = TYPE_NAMES.get(ty, ty); tm[k] += dt; cnt[k] += 1; el[k] += x.size
print(f"total {time.perf_counter() - T0:.1f} s")
for k, v in tm.most_common():
    print(f"{k:8s} {cnt[k]:5d} tensors {el[k] / 1e6:8.1f} M weights {v:7.1f} s {el[k] / v / 1e6:7.1f} M/s")
