"""Run every method on a GGUF file: size, ratio, speed, byte-exact round trip."""
import hashlib, json, sys, time
sys.path.insert(0, "src")
from codecs_gguf import METHODS, unpack
try:
    from gpu_codec import M6GPU
    from gpu2 import METHODS2
    from gpu3 import METHODS3
    METHODS = METHODS + [M6GPU()] + METHODS2 + METHODS3
except ImportError as e:
    print("GPU method unavailable:", e)

path = sys.argv[1] if len(sys.argv) > 1 else "models/Qwen3-0.6B-Q8_0.gguf"
only = sys.argv[2:]
data = open(path, "rb").read()
ref = hashlib.sha256(data).hexdigest()
mb = len(data) / 2**20
results = []
for m in METHODS:
    if only and not any(m.name.split()[0] == o for o in only):
        continue
    if hasattr(m, "timing"):
        m.decompress(m.compress(data))    # GPU methods: warm-up run (CUDA JIT / kernel cache), not timed
    t0 = time.perf_counter(); c = m.compress(data); t1 = time.perf_counter()
    d = m.decompress(c); t2 = time.perf_counter()
    ok = hashlib.sha256(d).hexdigest() == ref
    r = dict(method=m.name, size=len(c), saved_pct=100 * (1 - len(c) / len(data)),
             c_MBps=mb / (t1 - t0), d_MBps=mb / (t2 - t1), lossless=ok)
    try:
        r["chunks"] = {k: len(v) for k, v in unpack(c).items()}
    except Exception:
        pass
    results.append(r)
    print(f"{m.name:62s} {len(c):>11,} B  saved {r['saved_pct']:6.3f}%  "
          f"comp {r['c_MBps']:6.1f} MB/s  decomp {r['d_MBps']:6.1f} MB/s  lossless={ok}", flush=True)
    if getattr(m, "timing", None):
        r["timing"] = {k: {n: round(v, 3) for n, v in d.items()} for k, d in m.timing.items()}
        print("    ", r["timing"], flush=True)
    if "chunks" in r:
        print("    ", r["chunks"], flush=True)
json.dump(dict(file=path, size=len(data), results=results), open("results.json" if not only else "results_partial.json", "w"), indent=1)
