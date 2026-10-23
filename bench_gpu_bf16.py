"""M22/M23 (CPU) vs their GPU ports M24/M25. GPU methods get one untimed warm-up run."""
import hashlib, json, sys, time
sys.path.insert(0, "src")
from bf16codec import M22BF16
from bundle import M23Bundle
from gpu_bf16 import M24BF16GPU, M25BundleGPU

BF16, Q8 = "models/ref/Qwen3-0.6B-BF16.gguf", "models/Qwen3-0.6B-Q8_0.gguf"
which = sys.argv[1:] or ["M22", "M24", "M23", "M25"]
b = open(BF16, "rb").read(); q = open(Q8, "rb").read()
hb, hq = hashlib.sha256(b).hexdigest(), hashlib.sha256(q).hexdigest()
res = []


def report(name, raw, size, tc, td, ok, timing=None):
    mb = raw / 2**20
    r = dict(method=name, raw=raw, size=size, saved_pct=100 * (1 - size / raw), c_MBps=mb / tc, d_MBps=mb / td, lossless=ok)
    if timing:
        r["timing"] = timing
    res.append(r)
    print(f"{name:48s} {size:>13,} B  saved {r['saved_pct']:6.2f}%  comp {r['c_MBps']:7.1f} MB/s  decomp {r['d_MBps']:7.1f} MB/s  lossless={ok}", flush=True)
    if timing:
        print("    ", timing, flush=True)


for m in (M22BF16(), M24BF16GPU()):
    if m.name.split()[0] not in which:
        continue
    if isinstance(m, M24BF16GPU):
        m.decompress(m.compress(b))
    t0 = time.perf_counter(); c = m.compress(b); t1 = time.perf_counter(); d = m.decompress(c); t2 = time.perf_counter()
    report(m.name, len(b), len(c), t1 - t0, t2 - t1, hashlib.sha256(d).hexdigest() == hb, getattr(m, "timing", None))

for m in (M23Bundle(), M25BundleGPU()):
    if m.name.split()[0] not in which:
        continue
    if isinstance(m, M25BundleGPU):
        m.decompress_files(m.compress_files(b, [q]))
    t0 = time.perf_counter(); c = m.compress_files(b, [q]); t1 = time.perf_counter()
    b2, (q2,) = m.decompress_files(c); t2 = time.perf_counter()
    ok = hashlib.sha256(b2).hexdigest() == hb and hashlib.sha256(q2).hexdigest() == hq
    report(m.name + " [BF16+Q8_0]", len(b) + len(q), len(c), t1 - t0, t2 - t1, ok, getattr(m, "timing", None))

json.dump(res, open("results_bf16_gpu.json", "w"), indent=1)
