"""GPU re-quantisation of every (tensor, type) in the repository vs the real files: exactness and speed."""
import glob, os, sys, time, collections
sys.path.insert(0, "src")
import numpy as np, cupy as cp
from gguf_reader import GGUF, TYPE_NAMES
from requant import QFUN, load_imatrix
from gpuquant import GPUQuant

d = "models/unsloth"
base = open(os.path.join(d, "Qwen3-0.6B-BF16.gguf"), "rb").read()
bg = GGUF.from_bytes(base); by = {t.name: t for t in bg.tensors}
base_dev = cp.asarray(np.frombuffer(base, np.uint8))
imx = load_imatrix(os.path.join(d, "imatrix_unsloth.dat"))
g = GPUQuant()
want = {}                                     # (name, type) -> (file, tensor)
for p in sorted(glob.glob(os.path.join(d, "*.gguf"))):
    with open(p, "rb") as f:
        for t in GGUF.from_bytes(f.read(64 << 20)).tensors:
            if t.type in QFUN and (t.name, t.type) not in want:
                want[(t.name, t.type)] = (p, t)
bytype = collections.defaultdict(list)
for (n, ty), v in want.items():
    bytype[ty].append((n, v))
tot = 0; stat = collections.Counter(); bad = collections.Counter()
for ty, items in sorted(bytype.items()):
    cp.cuda.Device().synchronize(); t0 = time.perf_counter()
    xs = []
    for n, _ in items:
        bt = by[n]
        u = base_dev[bt.offset:bt.offset + bt.nbytes].view(cp.uint16).astype(cp.uint32) << 16
        xs.append(u.view(cp.float32).reshape(-1, bt.shape[0]))
    outs = g.quantize_batch(ty, xs, [imx.get(n) for n, _ in items])
    res = [cp.asnumpy(o) for o in outs]
    dt = time.perf_counter() - t0; tot += dt
    nb = 0
    for (n, (p, t)), r in zip(items, res):
        with open(p, "rb") as f:
            f.seek(t.offset); ref = np.frombuffer(f.read(t.nbytes), np.uint8)
        if (r != ref).any():
            nb += 1; bad[os.path.basename(p)] += 1
    el = sum(x.size for x in xs)
    print(f"{TYPE_NAMES[ty]:8s} {len(items):4d} tensors {el / 1e6:7.1f} M weights {dt:6.2f} s {el / dt / 1e6:8.1f} M/s  not exact: {nb}", flush=True)
    del xs, outs
print(f"total GPU requant {tot:.1f} s; non-exact by file: {dict(bad)}")
