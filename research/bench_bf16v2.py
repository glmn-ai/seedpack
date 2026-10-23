"""Round-trip + size/speed of R2 (bf16v2) on a BF16 GGUF. usage: bench_bf16v2.py [gguf] [budget bytes]"""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cupy as cp
from bf16v2 import BF16v2
path = sys.argv[1] if len(sys.argv) > 1 else "models/unsloth/Qwen3-0.6B-BF16.gguf"
budget = int(float(sys.argv[2])) if len(sys.argv) > 2 else 2_000_000_000
data = open(path, "rb").read()
c = BF16v2(budget=budget)
from bf16v2 import kern
for k in ("v2_hist", "v2_enc", "v2_dec", "row_hash"): kern(k)   # compile outside the timing
cp.get_default_memory_pool().free_all_blocks()
t0 = time.perf_counter(); arc = c.compress(data); t1 = time.perf_counter()
out = c.decompress(arc); t2 = time.perf_counter()
ok = bool(np.array_equal(out, np.frombuffer(data, np.uint8))); del out
peak = cp.get_default_memory_pool().total_bytes()
from seedpack.codecs_gguf import unpack
parts = {k: len(v) for k, v in unpack(arc).items()}
r = dict(method=c.name, file=path, raw=len(data), size=len(arc), pct=100 * len(arc) / len(data),
         vs_M24=len(arc) - 782650261, c_MBps=len(data) / (t1 - t0) / 1e6, d_MBps=len(data) / (t2 - t1) / 1e6,
         lossless=ok, parts=parts, timing=c.timing, gpu_pool_bytes=peak, budget=budget)
print(json.dumps(r, indent=1))
