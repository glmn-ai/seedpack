"""M28 (GPU re-quantisation): a whole model repository in one archive, speed and exactness.

usage: bench_repo3.py <repository dir> [archive path]
Bases = the BF16 GGUFs (model, mmproj); every other GGUF is derived from them; the rest
(imatrix, README, ...) is stored as is. Every file is checked by SHA-256 after decompression.
"""
import glob, hashlib, json, os, sys, time
sys.path.insert(0, ".")
from seedpack.repo3 import M28RepoGPU, M29RepoGPU
from seedpack.gpuquant import GPUQuant

d = sys.argv[1] if len(sys.argv) > 1 else "models/unsloth"
arc_path = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(os.path.normpath(d)) + ".seedpack"
paths = sorted(p for p in glob.glob(os.path.join(d, "*")) + glob.glob(os.path.join(d, ".*"))
               if os.path.isfile(p) and os.path.basename(p) != ".files")
base_paths = [p for p in paths if p.endswith(".gguf") and "BF16" in os.path.basename(p)]
gguf = [p for p in paths if p.endswith(".gguf") and p not in base_paths]
extras = {os.path.basename(p): p for p in paths if not p.endswith(".gguf")}
files = {os.path.basename(p): p for p in gguf}
allfiles = {os.path.basename(p): p for p in paths}
raw_total = sum(os.path.getsize(p) for p in paths)
bf16_size = sum(os.path.getsize(p) for p in base_paths)
print(f"{d}: {len(paths)} files, {raw_total / 1e9:.2f} GB; bases: {[os.path.basename(p) for p in base_paths]}")

ref = {}
for n, p in allfiles.items():
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    ref[n] = h.digest()

m = (M28RepoGPU if os.environ.get("METHOD") == "M28" else M29RepoGPU)(cache_bytes=(1 << 30) if bf16_size > 2e9 else (3 << 30)); m.verbose = True
print("method:", m.name)
tc = time.perf_counter(); m.gq = GPUQuant(); print(f"GPU quantiser build (cached by CuPy after the first run): {time.perf_counter() - tc:.1f} s")

t0 = time.perf_counter()
bases = {os.path.basename(p): p for p in base_paths}
arc = m.compress(bases, files, extras)
with open(arc_path, "wb") as f:
    f.write(arc)
t1 = time.perf_counter()
del bases
got = {}
from concurrent.futures import ThreadPoolExecutor
hp, pending = ThreadPoolExecutor(6), []


def sink(n, b):
    # SHA-256 in a thread pool (hashlib releases the GIL); outputs waiting are kept under 3 GB
    pending.append((n, hp.submit(lambda: hashlib.sha256(b).digest()), len(b)))
    while len(pending) > 1 and sum(x[2] for x in pending) > 3 << 30:
        k, f, _ = pending.pop(0); got[k] = f.result()


m.decompress(arc, sink)
for k, f, _ in pending:
    got[k] = f.result()
t2 = time.perf_counter()
bad = [n for n in allfiles if got.get(n) != ref[n]]

print(f"\n{'part':40s} {'in archive':>14s}")
for k, v in m.sizes.items():
    print(f"{k:40s} {v:>14,}")
print(f"\nrepository: {len(allfiles)} files, {raw_total:,} B ({raw_total / 1e9:.2f} GB)")
print(f"archive:    {len(arc):,} B ({len(arc) / 1e9:.3f} GB) = {100 * len(arc) / raw_total:.2f}% of the repository, "
      f"{100 * len(arc) / bf16_size:.1f}% of the BF16 file(s) ({bf16_size:,} B)")
print(f"archive <= BF16 file(s): {len(arc) <= bf16_size}")
print(f"all {len(allfiles)} files byte-exact after decompression: {not bad} {bad if bad else ''}")
print(f"tensors: {m.stats}")
print(f"compress {t1 - t0:.1f} s ({raw_total / (t1 - t0) / 1e6:.0f} MB/s), "
      f"decompress {t2 - t1:.1f} s ({raw_total / (t2 - t1) / 1e6:.0f} MB/s, incl. SHA-256 of every output file)")
json.dump(dict(method=m.name, repo=d, files=len(allfiles), raw_total=raw_total, archive=len(arc), bf16=bf16_size, parts=m.sizes,
               stats=m.stats, lossless=not bad, compress_s=t1 - t0, decompress_s=t2 - t1,
               compress_mbs=raw_total / (t1 - t0) / 1e6, decompress_mbs=raw_total / (t2 - t1) / 1e6),
          open(f"results_repo3_{os.path.basename(os.path.normpath(d))}.json", "w"), indent=1)

# decompression alone (output discarded), without the SHA-256 verification above
t3 = time.perf_counter(); m.decompress(arc, lambda n, b: None); t4 = time.perf_counter()
print(f"decompress without hashing: {t4 - t3:.1f} s ({raw_total / (t4 - t3) / 1e6:.0f} MB/s)")
r = json.load(open(f"results_repo3_{os.path.basename(os.path.normpath(d))}.json"))
r.update(decompress_nohash_s=t4 - t3, decompress_nohash_mbs=raw_total / (t4 - t3) / 1e6)
json.dump(r, open(f"results_repo3_{os.path.basename(os.path.normpath(d))}.json", "w"), indent=1)
