"""Whole unsloth/Qwen3-0.6B-GGUF repository (BF16 + 25 quants) in one M26 archive."""
import glob, hashlib, json, os, sys, time
sys.path.insert(0, ".")
from seedpack.gguf_reader import GGUF, TYPE_NAMES
from seedpack.repo import M26Repo

d = "models/unsloth"
base_path = os.path.join(d, "Qwen3-0.6B-BF16.gguf")
paths = sorted(p for p in glob.glob(os.path.join(d, "*.gguf")) if p != base_path)
base = open(base_path, "rb").read()
files = {os.path.basename(p).replace("Qwen3-0.6B-", "").replace(".gguf", ""): p for p in paths}
sizes = {k: os.path.getsize(p) for k, p in files.items()}
raw_total = len(base) + sum(sizes.values())
ref_hash = {k: hashlib.sha256(open(p, "rb").read()).digest() for k, p in files.items()}
got_hash = {}

m = M26Repo(); m.verbose = True
t0 = time.perf_counter(); arc = m.compress(base, files); t1 = time.perf_counter()
b2, _ = m.decompress(arc, sink=lambda n, a: got_hash.__setitem__(n, hashlib.sha256(a.tobytes()).digest()))
t2 = time.perf_counter()
ok_all = hashlib.sha256(b2.tobytes()).digest() == hashlib.sha256(base).digest()
rows = []
for name, p in files.items():
    ok = got_hash.get(name) == ref_hash[name]
    ok_all &= ok
    types = {}
    with open(p, "rb") as f:
        head = f.read(64 << 20)
    for t in GGUF.from_bytes(head).tensors:
        types[TYPE_NAMES.get(t.type, str(t.type))] = types.get(TYPE_NAMES.get(t.type, str(t.type)), 0) + t.nbytes
    main = ", ".join(k for k, _ in sorted(types.items(), key=lambda kv: -kv[1])[:3])
    rows.append(dict(file=name, size=sizes[name], archived=m.sizes[name], pct=100 * m.sizes[name] / sizes[name], lossless=ok, types=main))

print(f"{'file':14s} {'size MB':>9s} {'in archive MB':>14s} {'% of file':>9s}  types")
print(f"{'BF16 (base)':14s} {len(base) / 1e6:9.1f} {m.sizes['BF16 (base)'] / 1e6:14.1f} {100 * m.sizes['BF16 (base)'] / len(base):8.1f}%")
for r in sorted(rows, key=lambda r: r["pct"]):
    print(f"{r['file']:14s} {r['size'] / 1e6:9.1f} {r['archived'] / 1e6:14.1f} {r['pct']:8.1f}%  {r['types']}{'' if r['lossless'] else '  !! MISMATCH'}")
print(f"\nrepository: {len(files) + 1} files, {raw_total / 1e9:.2f} GB -> archive {len(arc) / 1e9:.3f} GB "
      f"({100 * len(arc) / raw_total:.1f}% of repo, {100 * len(arc) / len(base):.1f}% of the BF16 file)")
print(f"all files byte-exact: {ok_all}   compress {t1 - t0:.0f} s, decompress {t2 - t1:.0f} s")
print("prediction mismatches by type:")
for k, (n, mis, by) in sorted(m.stats.items()):
    print(f"  {TYPE_NAMES.get(k, k):8s} {n:>13,} values  {mis:>11,} mismatches ({100 * mis / n:.3f}%)")
json.dump(dict(raw_total=raw_total, archive=len(arc), base=len(base), base_archived=m.sizes["BF16 (base)"],
               files=rows, lossless=ok_all, compress_s=t1 - t0, decompress_s=t2 - t1,
               stats={TYPE_NAMES.get(k, k): v for k, v in m.stats.items()}),
          open("results_repo.json", "w"), indent=1)
