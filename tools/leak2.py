import os, sys
sys.path.insert(0, ".")
os.environ["SEEDPACK_MEMDEBUG"] = "1"
import numpy as np, cupy as cp
from seedpack import repo3
from seedpack.repo3 import M29RepoGPU, _keep_base, _mem, _release
d = "models/qwen35-2b"
m = M29RepoGPU(cache_bytes=1 << 30)
kept = []
for f in ("Qwen3.5-2B-BF16.gguf", "mmproj-BF16.gguf"):
    data = open(os.path.join(d, f), "rb").read(); kept.append(_keep_base(data)); del data
print("bases on gpu", _mem())
m._setup(kept, M29RepoGPU._imatrix({"imatrix": open(os.path.join(d, "imatrix_unsloth.gguf_file"), "rb").read()}))
print("setup", _mem())
orig = m.gq.quantize_tensors
def qt(ty, srcs, ims, v=0, **kw):
    r = orig(ty, srcs, ims, v, **kw); print(f"   type {ty} {sum(a.size for a, _ in srcs) / 1e6:.0f} M", _mem()); return r
m.gq.quantize_tensors = qt
for f in ["Qwen3.5-2B-IQ4_NL.gguf", "Qwen3.5-2B-Q8_0.gguf"]:
    data = bytearray(open(os.path.join(d, f), "rb").read()); print("read", f, _mem())
    m._residual(data); del data; _release(); print("done", _mem())
