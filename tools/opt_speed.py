"""Speed of the generated quantisers with NVVM -opt=0 vs -opt=3 (the latter is not bit-exact)."""
import os, sys, time
sys.path.insert(0, "src")
import numpy as np, cupy as cp
import gpuquant
from gguf_reader import GGUF
from requant import load_imatrix
d = "models/unsloth"
base = open(os.path.join(d, "Qwen3-0.6B-BF16.gguf"), "rb").read()
bg = GGUF.from_bytes(base)
ts = [t for t in bg.tensors if len(t.shape) == 2 and t.name.startswith("blk.")][:60]
bd = cp.asarray(np.frombuffer(base, np.uint8))
imx = load_imatrix(os.path.join(d, "imatrix_unsloth.dat"))
xs = [(bd[t.offset:t.offset + t.nbytes].view(cp.uint16).astype(cp.uint32) << 16).view(cp.float32).reshape(-1, t.shape[0]) for t in ts]
ims = [imx.get(t.name) for t in ts]
el = sum(x.size for x in xs)
mode = sys.argv[1]
g = gpuquant.GPUQuant.__new__(gpuquant.GPUQuant)
src = gpuquant.build_source()
cp.cuda.runtime.deviceSetLimit(0, 16384)
opts = ("--fmad=false", "--prec-div=true", "--prec-sqrt=true", "-std=c++17", "--diag-suppress=177,550,1650") + (("-Xnvvm", "-opt=0") if mode == "0" else ())
t0 = time.perf_counter(); g.mod = cp.RawModule(code=src, options=opts); g._upload_tables(); print(f"opt={mode} build {time.perf_counter() - t0:.1f} s")
for ty in (8, 12, 14, 16, 18, 19, 23):
    g.quantize_batch(ty, xs[:2], ims[:2]); cp.cuda.Device().synchronize()
    t0 = time.perf_counter(); g.quantize_batch(ty, xs, ims); cp.cuda.Device().synchronize(); dt = time.perf_counter() - t0
    print(f"  type {ty:2d}: {el / dt / 1e6:7.0f} M weights/s")
