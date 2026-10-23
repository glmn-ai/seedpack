"""GPU quantisers vs the CPU library: bit-exactness on real tensors of every type, and speed."""
import glob, os, sys, time, collections
sys.path.insert(0, "src")
import numpy as np, cupy as cp
from gguf_reader import GGUF, TYPE_NAMES
from requant import QFUN, quantize, load_imatrix
from repo2 import base_float_np
from gpuquant import GPUQuant

d = "models/unsloth"
base = open(os.path.join(d, "Qwen3-0.6B-BF16.gguf"), "rb").read()
bg = GGUF.from_bytes(base); by = {t.name: t for t in bg.tensors}
imx = load_imatrix(os.path.join(d, "imatrix_unsloth.dat"))
t0 = time.perf_counter(); g = GPUQuant(); print(f"compile+init {time.perf_counter() - t0:.1f} s")
names = ["blk.0.attn_q.weight", "blk.3.ffn_down.weight", "blk.10.attn_k.weight"]
for ty in QFUN:
    xs = [base_float_np(base, by[n]).reshape(-1, by[n].shape[0]) for n in names]
    ims = [imx.get(n) for n in names[:2]] + [None if ty < 16 else imx.get(names[2])]  # also one without imatrix
    gpu = g.quantize_batch(ty, [cp.asarray(x) for x in xs], ims)
    ok = []
    for x, im, gq in zip(xs, ims, gpu):
        c = quantize(ty, x, im)
        ok.append(int((c != cp.asnumpy(gq)).sum()))
    print(f"{TYPE_NAMES[ty]:8s} differing bytes: {ok}")
