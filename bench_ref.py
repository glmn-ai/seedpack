"""Reference-based methods (M12-M16): the receiver already has the reference GGUF(s)."""
import hashlib, json, sys, time
sys.path.insert(0, ".")
from seedpack.refcodec import M12RefXor, M13RefBase, M14RefQ4, M15RefBF16, M16RefMix

path = "models/Qwen3-0.6B-Q8_0.gguf"
BASE, Q4, BF16 = "models/ref/Qwen3-0.6B-Base.Q8_0.gguf", "models/ref/Qwen3-0.6B-Q4_K_M.gguf", "models/ref/Qwen3-0.6B-BF16.gguf"
METHODS = [M12RefXor(BASE), M13RefBase(BASE), M14RefQ4(Q4), M15RefBF16(BF16), M16RefMix([BASE, Q4])]
only = sys.argv[1:]
data = open(path, "rb").read()
ref = hashlib.sha256(data).hexdigest()
mb = len(data) / 2**20
results = []
for m in METHODS:
    if only and m.name.split()[0] not in only:
        continue
    m.refs()                                   # reference is already on the receiver: load outside timing
    t0 = time.perf_counter(); c = m.compress(data); t1 = time.perf_counter()
    d = m.decompress(c); t2 = time.perf_counter()
    ok = hashlib.sha256(d).hexdigest() == ref
    r = dict(method=m.name, refs=m.ref_paths, size=len(c), saved_pct=100 * (1 - len(c) / len(data)),
             c_MBps=mb / (t1 - t0), d_MBps=mb / (t2 - t1), lossless=ok)
    results.append(r)
    print(f"{m.name:58s} {len(c):>11,} B  saved {r['saved_pct']:6.2f}%  comp {r['c_MBps']:6.1f} MB/s  "
          f"decomp {r['d_MBps']:6.1f} MB/s  lossless={ok}", flush=True)
json.dump(dict(file=path, size=len(data), results=results), open("results_ref.json", "w"), indent=1)
