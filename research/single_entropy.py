"""Single quant file without a seed: how far below zstd can a structural model go?

Q4_K / Q6_K tensors: codes conditioned on (zero point of the sub-block in code units, column class),
sub-block scales conditioned on the same field one row above. Empirical conditional entropies
(an adaptive coder is within ~1% of them); side info (column classes) is counted.

usage: single_entropy.py FILE.gguf
"""
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import numpy as np
import zstandard

from qunpack import q4_k, q6_k  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from gguf_reader import GGUF  # noqa: E402


def H(counts):
    c = np.asarray(counts, np.float64)
    tot = c.sum(-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(-np.nansum(np.where(c > 0, c * np.log2(c / tot), 0))) / 8


g = GGUF(sys.argv[1], light=True)
z = zstandard.ZstdCompressor(level=19)
res = defaultdict(lambda: np.zeros(4))           # bytes, zstd, model codes, model scales
for t in g.tensors:
    if t.type not in (12, 14):
        continue
    cols = t.shape[0]; rows = t.nbytes * (256 if t.type == 12 else 256) // (144 if t.type == 12 else 210) // cols
    bs = 144 if t.type == 12 else 210
    b = np.frombuffer(g.data, np.uint8, t.nbytes, t.offset).reshape(-1, bs)
    q, a, c, sf = (q4_k if t.type == 12 else q6_k)(b)
    with np.errstate(divide="ignore", invalid="ignore"):
        zp = np.where(a != 0, -c / a, 0)                     # zero point in code units
    nb_row = cols // 256
    # column class: mean |q - zero point| per column (8 classes, stored as side info)
    qr_ = q.reshape(rows, nb_row * 256).astype(np.float32); zr = zp.reshape(rows, nb_row * 256)
    dev = np.abs(qr_ - zr).mean(0)
    cls = np.clip(np.digitize(dev, np.quantile(dev, np.linspace(0, 1, 9)[1:-1])), 0, 7)
    ctx_z = np.clip(np.rint(zr * 2), 0, 2 * (16 if t.type == 12 else 64)).astype(np.int64)
    key = (np.broadcast_to(cls[None, :], qr_.shape).astype(np.int64) * 256 + ctx_z) * 64 + q.reshape(rows, -1)
    cnt = np.bincount(key.ravel(), minlength=8 * 256 * 64).reshape(-1, 64)
    hc = H(cnt) + cols * 3 / 8
    # scale fields: bytes conditioned on the same byte one row above
    s_cur = sf.reshape(rows, nb_row, -1)
    above = np.concatenate([np.zeros_like(s_cur[:1]), s_cur[:-1]], 0)
    pos = np.broadcast_to(np.arange(sf.shape[1])[None, None, :], s_cur.shape)
    keys = (pos.astype(np.int64) * 256 + above) * 256 + s_cur
    hs = H(np.bincount(keys.ravel(), minlength=sf.shape[1] * 65536).reshape(-1, 256))
    r = res["Q4_K" if t.type == 12 else "Q6_K"]
    r += [t.nbytes, len(z.compress(b.tobytes())), hc, hs]
print(os.path.basename(sys.argv[1]))
for k, (n, zs, hc, hs) in res.items():
    print(f"{k}: {n/1e6:.1f} MB  zstd-19 {100*zs/n:.2f}%   model {100*(hc+hs)/n:.2f}% (codes {100*hc/n:.2f}%, scales {100*hs/n:.2f}%)")
