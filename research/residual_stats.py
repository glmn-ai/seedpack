"""Where the bytes of a seed-based quant archive go: per tensor type, how much of the residual
is non-zero (blocks, bytes) and what zstd makes of it.

usage: residual_stats.py QUANT_DIR [IMATRIX_FILE] [--level N]
The seed is chosen as Seedpack does (most bits per weight); every other GGUF is predicted from
it exactly as repo3.M29RepoGPU._residual does (duplicates across files included).
"""
import glob
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import zstandard

from seedpack.gguf_reader import GGUF, TYPE_NAMES, BLOCK
from seedpack.repo3 import M29RepoGPU, _keep_base, _read_file, _gguf_tensors
from seedpack.requant import load_imatrix
from seedpack.seed import bits_per_weight

args = [a for a in sys.argv[1:] if not a.startswith("--")]
level = int(sys.argv[sys.argv.index("--level") + 1]) if "--level" in sys.argv else 7
d = args[0]
paths = sorted(glob.glob(os.path.join(d, "*.gguf")))
seed = max(paths, key=lambda p: bits_per_weight(_gguf_tensors(p)))
im = load_imatrix(open(args[1], "rb").read()) if len(args) > 1 else {}
m = M29RepoGPU()
m._setup([_keep_base(_read_file(seed))], im)
m.seen_t, m.keep_until, m.live = {}, {}, []
z = zstandard.ZstdCompressor(level=level)
by_type = defaultdict(lambda: np.zeros(5))      # bytes, nonzero bytes, blocks, nonzero blocks, zstd bytes
by_file = {}
print(f"seed: {os.path.basename(seed)}   imatrix: {'yes' if im else 'no'}")
for fi, p in enumerate(x for x in paths if x != seed):
    data = _read_file(p)
    plan, b = m._residual(data, fi=fi)
    view = np.frombuffer(data, np.uint8)
    kinds = {e[0]: e for e in plan}
    g = GGUF.from_bytes(data, light=True)
    tot_z = len(z.compress(bytes(data)))
    by_file[os.path.basename(p)] = (len(data), tot_z)
    for t in g.tensors:
        r = view[t.offset:t.offset + t.nbytes]
        e = kinds.get(t.name)
        kind = "dup" if e and e[1] == "dup" else ("raw" if e is None else TYPE_NAMES.get(t.type, str(t.type)))
        be, bb = BLOCK[t.type]
        blocks = r[: len(r) // bb * bb].reshape(-1, bb)
        nzb = int(np.count_nonzero(blocks.any(1)))
        by_type[kind] += [t.nbytes, np.count_nonzero(r), len(blocks), nzb, len(z.compress(r.tobytes())) if r.any() else 0]
    m._release() if hasattr(m, "_release") else None
print(f"\n{'type':8s} {'GB':>7s} {'nz bytes':>9s} {'nz blocks':>10s} {'zstd MB':>9s} {'% of type':>9s}")
for k, (n, nz, nb, nzb, zb) in sorted(by_type.items(), key=lambda kv: -kv[1][4]):
    print(f"{k:8s} {n/1e9:7.3f} {100*nz/max(n,1):8.2f}% {100*nzb/max(nb,1):9.2f}% {zb/1e6:9.1f} {100*zb/max(n,1):8.2f}%")
tot = sum(v[0] for v in by_file.values()); totz = sum(v[1] for v in by_file.values())
print(f"\nresiduals of {len(by_file)} files: {tot/1e9:.2f} GB -> {totz/1e6:.0f} MB with zstd -{level} ({100*totz/tot:.2f}%)")
for n, (s, zs) in sorted(by_file.items(), key=lambda kv: -kv[1][1])[:8]:
    print(f"   {n:32s} {s/1e6:8.1f} MB -> {zs/1e6:7.1f} MB ({100*zs/s:5.1f}%)")
