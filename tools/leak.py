import glob, os, sys, tracemalloc, gc
sys.path.insert(0, ".")
import numpy as np, cupy as cp, psutil
from seedpack.repo3 import M28RepoGPU
d = "models/qwen35-08b"
P = lambda: psutil.Process().memory_info().private / 1e9
m = M28RepoGPU(cache_bytes=0)
bases = [open(os.path.join(d, f), "rb").read() for f in ("Qwen3.5-0.8B-BF16.gguf", "mmproj-BF16.gguf")]
print("after read bases", P())
m._setup([(b, cp.asarray(np.frombuffer(b, np.uint8))) for b in bases], M28RepoGPU._imatrix({"imatrix": open(os.path.join(d, "imatrix_unsloth.gguf_file"), "rb").read()}))
print("after setup", P())
tracemalloc.start()
for f in ["Qwen3.5-0.8B-Q4_0.gguf", "Qwen3.5-0.8B-Q4_1.gguf", "Qwen3.5-0.8B-Q6_K.gguf", "Qwen3.5-0.8B-Q8_0.gguf"]:
    data = bytearray(open(os.path.join(d, f), "rb").read())
    m._residual(data); del data; gc.collect(); import repo3; repo3._release()
    cur, peak = tracemalloc.get_traced_memory()
    print(f, "private", round(P(), 2), "traced", round(cur / 1e9, 2), "peak", round(peak / 1e9, 2))
snap = tracemalloc.take_snapshot()
for s in snap.statistics("lineno")[:6]: print(s)
