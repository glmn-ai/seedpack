"""No-seed QR coding of a single quant file: round trip + size per type vs zstd -19."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, cupy as cp, zstandard
from seedpack import qr
qr.STATS = {}
from seedpack.gguf_reader import GGUF, TYPE_NAMES
g = GGUF(sys.argv[1], light=True)
z = zstandard.ZstdCompressor(level=19)
by = {}
for t in g.tensors:
    if t.type in qr.QR_TYPES and (t.type != 8 or "--q8" in sys.argv):
        by.setdefault(t.type, []).append(t)
tot_n = tot_z = tot_q = 0
for T, ts in by.items():
    bl = [np.frombuffer(g.data, np.uint8, t.nbytes, t.offset) for t in ts]
    t0 = time.time(); buf = qr.self_encode(T, [(b, t.shape[0]) for b, t in zip(bl, ts)]); t1 = time.time()
    out = qr.self_decode(T, buf); t2 = time.time()
    assert all(np.array_equal(o.get(), b) for o, b in zip(out, bl)), "round trip FAILED"
    n = sum(len(b) for b in bl); zs = len(z.compress(b"".join(b.tobytes() for b in bl)))
    tot_n += n; tot_z += zs; tot_q += len(buf)
    print(f"{TYPE_NAMES[T]:7s} {n/1e6:7.1f} MB  zstd-19 {100*zs/n:6.2f}%  self-QR {100*len(buf)/n:6.2f}%   enc {t1-t0:.1f}s dec {t2-t1:.1f}s")
print(f"all QR-able tensors: {tot_n/1e6:.1f} MB of {len(g.data)/1e6:.1f} MB file: zstd {100*tot_z/tot_n:.2f}%  self-QR {100*tot_q/tot_n:.2f}%")
for (T, k), v in sorted(qr.STATS.items()):
    print(f"   {TYPE_NAMES[T]} {k}: {v/1e6:.2f} MB")
