"""GPU quantisation time per ggml type for the unique (tensor, type) pairs of a repository."""
import glob, os, sys, time, collections
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, cupy as cp
from seedpack.gguf_reader import GGUF, TYPE_NAMES
from seedpack.gpuquant import GPUQuant
from seedpack.requant import QFUN, load_imatrix
d = sys.argv[1] if len(sys.argv) > 1 else "models/unsloth"
base = open(glob.glob(os.path.join(d, "*BF16*.gguf"))[0], "rb").read()
bg = GGUF.from_bytes(base); by = {t.name: t for t in bg.tensors}
dev = cp.asarray(np.frombuffer(base, np.uint8))
imx = load_imatrix(glob.glob(os.path.join(d, "imatrix*"))[0])
want = collections.defaultdict(dict)
for p in sorted(glob.glob(os.path.join(d, "*.gguf"))):
    if "BF16" in p: continue
    with open(p, "rb") as f: head = f.read(16 << 20)
    for t in GGUF.from_bytes(head).tensors:
        if t.type in QFUN: want[t.type][t.name] = t
gq = GPUQuant()
tot = 0
for ty, ts in want.items():
    srcs = [(dev[by[n].offset:by[n].offset + by[n].nbytes].view(cp.uint16), t.shape[0]) for n, t in ts.items()]
    nw = sum(s[0].size for s in srcs)
    cp.cuda.Device().synchronize(); t0 = time.perf_counter()
    gq.quantize_tensors(ty, srcs, [imx.get(n) for n in ts], 0, max_weights=1 << 26)
    cp.cuda.Device().synchronize(); dt = time.perf_counter() - t0; tot += dt
    print(f"{TYPE_NAMES[ty]:8s} {len(ts):4d} tensors {nw / 1e6:7.1f} Mw  {dt:6.2f}s  {nw / dt / 1e6:8.1f} Mw/s", flush=True)
print(f"total {tot:.2f}s")
