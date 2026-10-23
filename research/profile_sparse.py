"""Stage times of R3 decompression (wall-clock accumulators, main thread)."""
import glob, hashlib, os, sys, time, collections
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cupy as cp
import repo_sparse
from repo_sparse import R3RepoSparse, R2Base
from seedpack.gpuquant import GPUQuant
T = collections.defaultdict(float)
def wrap(obj, name, label):
    f = getattr(obj, name)
    def g(*a, **k):
        t = time.perf_counter()
        try: return f(*a, **k)
        finally: T[label] += time.perf_counter() - t
    setattr(obj, name, g)
d = "models/unsloth"
paths = sorted(p for p in glob.glob(os.path.join(d, "*")) + glob.glob(os.path.join(d, ".*")) if os.path.isfile(p) and os.path.basename(p) != ".files")
base_paths = [p for p in paths if p.endswith(".gguf") and "BF16" in os.path.basename(p)]
files = {os.path.basename(p): p for p in paths if p.endswith(".gguf") and p not in base_paths}
extras = {os.path.basename(p): p for p in paths if not p.endswith(".gguf")}
m = R3RepoSparse(base_codec=R2Base(), cache_bytes=int(float(sys.argv[1]) * (1 << 30)) if len(sys.argv) > 1 else 1 << 30)
m.gq = GPUQuant()
wrap(m, "_predict", "predict (GPU quant + D2H + cache)")
wrap(m.gq, "quantize_tensors", "  of which quantize_tensors")
wrap(m.base_codec, "compress_dev", "base compress")
wrap(m.base_codec, "decompress_dev", "base decompress")
wrap(m, "_residual2", "residual2 (parse+predict+xor)")
t0 = time.perf_counter()
bases = {os.path.basename(p): open(p, "rb").read() for p in base_paths}
arc = m.compress(bases, files, extras); t1 = time.perf_counter(); del bases
print(f"compress {t1 - t0:.1f}s"); [print(f"  {k:40s} {v:6.2f}") for k, v in T.items()]; T.clear()
def sink(n, b):
    t = time.perf_counter(); hashlib.sha256(b).digest(); T["sink: sha256"] += time.perf_counter() - t
t1 = time.perf_counter(); m.decompress(arc, sink); t2 = time.perf_counter()
print(f"decompress {t2 - t1:.1f}s"); [print(f"  {k:40s} {v:6.2f}") for k, v in T.items()]
