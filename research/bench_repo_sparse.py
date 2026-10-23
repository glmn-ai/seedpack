"""R3 vs M28-style residual on a repository, same base codec (R2) and same cache, SHA-256 checked.
usage: bench_repo_sparse.py [repo dir] [mode: sparse|m28] [cache GB]"""
import glob, hashlib, json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import cupy as cp
from repo_sparse import R3RepoSparse, R2Base
from gpuquant import GPUQuant
d = sys.argv[1] if len(sys.argv) > 1 else "models/unsloth"
mode = sys.argv[2] if len(sys.argv) > 2 else "sparse"
cache = float(sys.argv[3]) if len(sys.argv) > 3 else 2
paths = sorted(p for p in glob.glob(os.path.join(d, "*")) + glob.glob(os.path.join(d, ".*")) if os.path.isfile(p) and os.path.basename(p) != ".files")
base_paths = [p for p in paths if p.endswith(".gguf") and "BF16" in os.path.basename(p)]
files = {os.path.basename(p): p for p in paths if p.endswith(".gguf") and p not in base_paths}
extras = {os.path.basename(p): p for p in paths if not p.endswith(".gguf")}
raw_total = sum(os.path.getsize(p) for p in paths)
ref = {}
for p in paths:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""): h.update(c)
    ref[os.path.basename(p)] = h.digest()
m = R3RepoSparse(base_codec=R2Base(), sparse=(mode == "sparse"), extras_r5=(mode == "sparse"), cache_bytes=int(cache * (1 << 30)))
m.gq = GPUQuant()
from bf16v2 import kern
for k in ("v2_hist", "v2_enc", "v2_dec", "row_hash"): kern(k)
t0 = time.perf_counter()
bases = {os.path.basename(p): open(p, "rb").read() for p in base_paths}
arc = m.compress(bases, files, extras)
t1 = time.perf_counter()
del bases
cp.get_default_memory_pool().free_all_blocks()
got = {}
t2a = time.perf_counter()
hs = [0.0]
def sink(n, b):
    t = time.perf_counter(); got[n] = hashlib.sha256(b).digest(); hs[0] += time.perf_counter() - t
m.decompress(arc, sink)
t2 = time.perf_counter()
bad = [n for n in ref if got.get(n) != ref[n]]
r = dict(mode=mode, repo=d, raw_total=raw_total, archive=len(arc), lossless=not bad, bad=bad,
         compress_s=round(t1 - t0, 2), decompress_s=round(t2 - t2a, 2), decompress_excl_sha_s=round(t2 - t2a - hs[0], 2), parts=m.sizes, stats=m.stats, cache_gb=cache)
print(json.dumps(r, indent=1))
json.dump(r, open(os.path.join(os.path.dirname(__file__), f"result_repo_{mode}_{os.path.basename(os.path.normpath(d))}.json"), "w"), indent=1)
