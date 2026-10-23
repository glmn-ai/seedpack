"""M27: the whole unsloth/Qwen3-0.6B-GGUF repository (all 32 files) in one archive."""
import glob, hashlib, json, os, sys, time
sys.path.insert(0, ".")
from seedpack.repo2 import M27RepoRequant

d = "models/unsloth"
base_path = os.path.join(d, "Qwen3-0.6B-BF16.gguf")
gguf = sorted(p for p in glob.glob(os.path.join(d, "*.gguf")) if p != base_path)
extras = {os.path.basename(p): p for p in glob.glob(os.path.join(d, "*")) + glob.glob(os.path.join(d, ".*"))
          if os.path.isfile(p) and not p.endswith(".gguf")}
files = {os.path.basename(p): p for p in gguf}
allfiles = {"BF16": base_path, **extras, **files}
ref = {}
for n, p in allfiles.items():
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    ref[n] = h.digest()
raw_total = sum(os.path.getsize(p) for p in allfiles.values())
bf16_size = os.path.getsize(base_path)

base = open(base_path, "rb").read()
m = M27RepoRequant(); m.verbose = True
t0 = time.perf_counter(); arc = m.compress(base, files, extras); t1 = time.perf_counter()
del base
with open("qwen3-0.6b-repo.varc", "wb") as f:
    f.write(arc)
got = {}
m.decompress(arc, lambda n, b: got.__setitem__(n, hashlib.sha256(b).digest()))
t2 = time.perf_counter()
bad = [n for n in allfiles if got.get(n) != ref[n]]

print(f"\n{'part':28s} {'in archive':>14s}")
for k, v in m.sizes.items():
    print(f"{k:28s} {v:>14,}")
print(f"\nrepository: {len(allfiles)} files, {raw_total:,} B ({raw_total / 1e9:.2f} GB)")
print(f"archive:    {len(arc):,} B ({len(arc) / 1e9:.3f} GB) = {100 * len(arc) / raw_total:.2f}% of the repository, "
      f"{100 * len(arc) / bf16_size:.1f}% of the BF16 file ({bf16_size:,} B)")
print(f"archive <= BF16 file: {len(arc) <= bf16_size}")
print(f"all {len(allfiles)} files byte-exact after decompression: {not bad} {bad if bad else ''}")
print(f"tensors: {m.stats}")
print(f"compress {t1 - t0:.0f} s, decompress {t2 - t1:.0f} s")
json.dump(dict(files=len(allfiles), raw_total=raw_total, archive=len(arc), bf16=bf16_size, parts=m.sizes,
               stats=m.stats, lossless=not bad, compress_s=t1 - t0, decompress_s=t2 - t1),
          open("results_repo2.json", "w"), indent=1)
