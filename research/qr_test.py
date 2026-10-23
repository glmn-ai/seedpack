"""QR coder round trip + size vs XOR/zstd on real tensors.

usage: qr_test.py QUANT_DIR FILE_PATTERN... [--imatrix FILE]
"""
import glob
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import cupy as cp
import numpy as np
import zstandard

from seedpack import qr
from seedpack.gguf_reader import GGUF, TYPE_NAMES
from seedpack.repo3 import M29RepoGPU, _keep_base, _read_file, _gguf_tensors
from seedpack.requant import load_imatrix
from seedpack.seed import bits_per_weight

args = [a for i, a in enumerate(sys.argv[1:], 1) if not a.startswith("--") and not sys.argv[i - 1].startswith("--")]
imf = sys.argv[sys.argv.index("--imatrix") + 1] if "--imatrix" in sys.argv else None
d, pats = args[0], args[1:]
paths = sorted(glob.glob(os.path.join(d, "*.gguf")))
seed = max(paths, key=lambda p: bits_per_weight(_gguf_tensors(p)))
im = load_imatrix(open(imf, "rb").read()) if imf else {}
m = M29RepoGPU()
m._setup([_keep_base(_read_file(seed))], im)
z = zstandard.ZstdCompressor(level=7)
st = defaultdict(lambda: np.zeros(4))
tenc = tdec = 0.0
for p in [x for x in paths if any(s in os.path.basename(x) for s in pats) and x != seed]:
    g = GGUF(p, light=True)
    by = defaultdict(list)
    for t in g.tensors:
        bt = m.by_name.get(t.name)
        if t.type in qr.QR_TYPES and bt is not None and bt.type in qr.SEED_TYPES:
            by[t.type].append((t, bt))
    for T, ts in by.items():
        preds = m.gq.quantize_tensors(T, [(m._src(bt), t.shape[0]) for t, bt in ts],
                                      [im.get(t.name) if t.name in im else np.ones(t.shape[0], np.float32) for t, _ in ts])
        items = [(np.frombuffer(g.data, np.uint8, t.nbytes, t.offset), pr, bt.offset, bt.type) for (t, bt), pr in zip(ts, preds)]
        for bi in qr.batches(T, [t.nbytes for t, _ in ts]):
            it = [items[i] for i in bi]
            cp.cuda.Device().synchronize(); t0 = time.perf_counter()
            buf = qr.encode(T, it, m.base_dev)
            cp.cuda.Device().synchronize(); t1 = time.perf_counter()
            out = qr.decode(T, buf, [(pr, so, sty) for _, pr, so, sty in it], m.base_dev).get()
            cp.cuda.Device().synchronize(); t2 = time.perf_counter()
            tenc += t1 - t0; tdec += t2 - t1
            ref = np.concatenate([f for f, _, _, _ in it])
            assert np.array_equal(out, ref), f"{TYPE_NAMES[T]} round trip FAILED"
            xz = sum(len(z.compress(np.bitwise_xor(f, pr).tobytes())) for f, pr, _, _ in it)
            st[TYPE_NAMES[T]] += [len(ref), xz, len(buf), 0]
print(f"seed {os.path.basename(seed)}, imatrix {'yes' if im else 'no'}; all round trips exact")
print(f"{'type':7s} {'MB':>7s} {'XOR+zstd':>9s} {'QR':>8s} {'gain':>6s}")
for k, (n, xz, q, _) in sorted(st.items(), key=lambda kv: -kv[1][1]):
    print(f"{k:7s} {n/1e6:7.1f} {xz/1e6:9.2f} {q/1e6:8.2f} {xz/max(q,1):5.2f}x")
tot = sum(v[0] for v in st.values())
print(f"total {tot/1e6:.0f} MB: XOR+zstd {sum(v[1] for v in st.values())/1e6:.1f} MB, QR {sum(v[2] for v in st.values())/1e6:.1f} MB; "
      f"QR encode {tot/tenc/1e9:.2f} GB/s, decode {tot/tdec/1e9:.2f} GB/s")
