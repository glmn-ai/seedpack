"""Where does M28 spend its time? Wraps the main steps with wall-clock accumulators.
usage: profile_m28.py [repo dir]"""
import glob, os, sys, time, threading, collections
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import cupy as cp
import repo3, gpuquant, gguf_reader
from repo3 import M28RepoGPU
from gpuquant import GPUQuant

T = collections.defaultdict(float); lock = threading.Lock()


def wrap(obj, name, label):
    f = getattr(obj, name)

    def g(*a, **k):
        t = time.perf_counter()
        try:
            return f(*a, **k)
        finally:
            with lock:
                T[label] += time.perf_counter() - t
    setattr(obj, name, g)


d = sys.argv[1] if len(sys.argv) > 1 else "models/unsloth"
paths = sorted(p for p in glob.glob(os.path.join(d, "*")) + glob.glob(os.path.join(d, ".*")) if os.path.isfile(p) and os.path.basename(p) != ".files")
base_paths = [p for p in paths if p.endswith(".gguf") and "BF16" in os.path.basename(p)]
files = {os.path.basename(p): p for p in paths if p.endswith(".gguf") and p not in base_paths}
extras = {os.path.basename(p): p for p in paths if not p.endswith(".gguf")}

m = M28RepoGPU(cache_bytes=6 << 30)
m.gq = GPUQuant()


class RawBase:            # stand-in for M24 (keeps GPU memory low; M24 itself is benchmarked separately)
    def compress_dev(self, data, dev=None):
        return {"raw": bytes(data)}

    def decompress_dev(self, ch):
        return cp.asarray(np.frombuffer(ch["raw"], np.uint8))


m.base_codec = RawBase()
_qt = m.gq.quantize_tensors
m.gq.quantize_tensors = lambda *a, **k: _qt(*a, max_weights=1 << 26, **k)
wrap(m.gq, "quantize_tensors", "gpu quantize (incl. D2H)")
wrap(m, "_residual", "residual total (predict+compare+xor)")
wrap(m.base_codec, "compress_dev", "M24 compress")
wrap(m.base_codec, "decompress_dev", "M24 decompress")
wrap(repo3.GGUF, "from_bytes", "GGUF parse")
import zstandard
_c = zstandard.ZstdCompressor.compress; _d = zstandard.ZstdDecompressor.decompress


def zc(self, *a, **k):
    t = time.perf_counter(); r = _c(self, *a, **k)
    with lock: T["zstd compress (thread time)"] += time.perf_counter() - t
    return r


def zd(self, *a, **k):
    t = time.perf_counter(); r = _d(self, *a, **k)
    with lock: T["zstd decompress (thread time)"] += time.perf_counter() - t
    return r


zstandard.ZstdCompressor.compress = zc; zstandard.ZstdDecompressor.decompress = zd
t0 = time.perf_counter()
bases = {os.path.basename(p): open(p, "rb").read() for p in base_paths}
arc = m.compress(bases, files, extras)
t1 = time.perf_counter()
print(f"compress {t1 - t0:.1f}s, archive {len(arc):,}  peak GPU pool {cp.get_default_memory_pool().total_bytes() / 1e9:.2f} GB")
for k, v in T.items(): print(f"  {k:40s} {v:7.2f}s")
T.clear()
n = [0]
m.decompress(arc, lambda name, b: n.__setitem__(0, n[0] + len(b)))
t2 = time.perf_counter()
print(f"decompress {t2 - t1:.1f}s ({n[0]:,} B out)")
for k, v in T.items(): print(f"  {k:40s} {v:7.2f}s")
