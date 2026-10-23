"""Per-file exactness of M28's predictions on the complete GGUFs of a repository directory:
check_repo_exact.py <dir> [file substrings...]"""
import glob, os, sys, collections
sys.path.insert(0, "src")
import numpy as np, cupy as cp
from repo3 import M28RepoGPU, _par_xor
from gguf_reader import GGUF, TYPE_NAMES
d = sys.argv[1]; only = sys.argv[2:]
paths = sorted(glob.glob(os.path.join(d, "*.gguf")))
bases = [p for p in paths if "BF16" in os.path.basename(p)]
files = [p for p in paths if p not in bases and (not only or any(s in p for s in only))]
m = M28RepoGPU()
bdata = [open(p, "rb").read() for p in bases]
imx = M28RepoGPU._imatrix({os.path.basename(p): open(p, "rb").read() for p in glob.glob(os.path.join(d, "imatrix*"))})
m._setup([(b, cp.asarray(np.frombuffer(b, np.uint8))) for b in bdata], imx)
for p in files:
    data = bytearray(open(p, "rb").read())
    before = dict(m.stats)
    plan, b = m._residual(data)
    st = {k: m.stats[k] - before[k] for k in m.stats}
    view = np.frombuffer(data, np.uint8)
    tens = {t.name: t for t in GGUF.from_bytes(data, light=True).tensors}
    bad = collections.Counter()
    for n, k, v in plan:
        t = tens[n]
        if view[t.offset:t.offset + t.nbytes].any():
            bad[TYPE_NAMES.get(t.type, t.type)] += 1
    raw = [f"{n}:{TYPE_NAMES.get(t.type, t.type)}" for n, t in tens.items() if n not in {x[0] for x in plan}]
    print(f"{os.path.basename(p):32s} base={os.path.basename(bases[b]):26s} {st}  non-exact by type: {dict(bad)}  raw: {raw[:4]}{'...' if len(raw) > 4 else ''}", flush=True)
