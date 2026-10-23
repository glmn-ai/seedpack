"""Fit an importance-matrix vector from quantised files alone.

For one layer group (tensors sharing an input: attn_q/k/v, ffn_gate/up, ...), the K-/IQ4 quants of
the given files are re-quantised from the seed with a candidate vector w; the score is the number
of weights whose dequantised value matches the file's. Coordinate search: in each pass one column
per 32-column group is multiplied by a factor in all groups at once (sub-block decisions are
independent), and kept in the groups whose score improved.

usage: imatrix_fit.py QUANT_DIR IMATRIX_FILE LAYER_GROUP [--rounds N] [--rows FRACTION]
  LAYER_GROUP e.g. blk.3.attn_q (shares with attn_k, attn_v) or blk.3.ffn_gate
"""
import glob
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import cupy as cp
import numpy as np

import qr

from gguf_reader import GGUF
from repo3 import M29RepoGPU, _keep_base, _read_file, _gguf_tensors
from requant import load_imatrix
from seed import bits_per_weight

SHARE = {"attn_q": ["attn_q", "attn_k", "attn_v"], "ffn_gate": ["ffn_gate", "ffn_up"],
         "attn_output": ["attn_output"], "ffn_down": ["ffn_down"]}
FIT_TYPES = {10, 11, 12, 13, 14, 23}           # Q2_K..Q6_K, IQ4_XS


def opt(n, d):
    return sys.argv[sys.argv.index(n) + 1] if n in sys.argv else d


d, imf, grp = sys.argv[1], sys.argv[2], sys.argv[3]
rounds, rowfrac = int(opt("--rounds", 2)), float(opt("--rows", 1.0))
layer, kind = grp.rsplit(".", 1)
names = [f"{layer}.{k}.weight" for k in SHARE[kind]]
paths = sorted(glob.glob(os.path.join(d, "*.gguf")))
seed = max(paths, key=lambda p: bits_per_weight(_gguf_tensors(p)))
true_im = load_imatrix(open(imf, "rb").read())
m = M29RepoGPU()
m._setup([_keep_base(_read_file(seed))], {})
gq = m.gq
obs = []                                        # (type, rows, cols, seed values, file dequant)
for p in paths:
    if p == seed:
        continue
    g = GGUF(p, light=True)
    for t in g.tensors:
        if t.name in names and t.type in FIT_TYPES:
            cols = t.shape[0]; rows = int(np.prod(t.shape)) // cols
            nr = max(1, int(rows * rowfrac)) // 8 * 8 or rows
            src = m._src(m.by_name[t.name])
            x = (src.f32() if hasattr(src, "f32") else gq.dequantize(1, src, cols) if False else None)
            if x is None:
                xx = cp.empty(rows * cols, cp.float32)
                from gpuquant import _to_f32_into
                _to_f32_into(src, xx); x = xx
            x = x[: nr * cols]
            fb = cp.asarray(np.frombuffer(g.data, np.uint8, t.nbytes, t.offset)[: t.nbytes // rows * nr])
            obs.append((t.type, nr, cols, x, fb))
cols = obs[0][2]
print(f"{grp}: {len(obs)} observations ({sorted({o[0] for o in obs})}), {cols} columns, seed {os.path.basename(seed)}")


def score(w):
    """per-column count of matching weights over all observations"""
    s = cp.zeros(cols, cp.float64)
    for T, nr, c, x, fd in obs:
        pr = cp.asarray(gq.quantize_tensors(T, [(x, c)], [w])[0])
        s += qr.weight_equal(T, fd, pr, nr * c).reshape(nr, c).sum(0)
    return s


total = sum(o[1] * o[2] for o in obs)
s_true = score(true_im[names[0]]).sum()
w = np.ones(cols, np.float32)
s = score(w)
print(f"match: unit weights {100*s.sum()/total:.2f}%, true imatrix {100*s_true/total:.2f}%")
NORM = {"attn_q": "attn_norm", "ffn_gate": "ffn_norm"}
if "--norm" in sys.argv and kind in NORM:
    # RMSNorm inputs: imatrix ~ norm_weight^2 x (per-channel energy of the normalised stream)
    g0 = GGUF(seed, light=True)
    nt = [t for t in g0.tensors if t.name == f"{layer}.{NORM[kind]}.weight"][0]
    nw = np.frombuffer(g0.data, np.float32, cols, nt.offset).astype(np.float64)
    w = (nw * nw + 1e-12).astype(np.float32)
    s = score(w)
    print(f"match with norm^2 init: {100*s.sum()/total:.2f}%")
t0 = time.time()
G = 32
for r in range(rounds):
    for f in (8.0, 0.125, 2.0, 0.5, 1.3, 0.77):
        for ph in range(G):
            w2 = w.copy(); w2[ph::G] *= f
            s2 = score(w2)
            g1 = s.reshape(-1, G).sum(1); g2 = s2.reshape(-1, G).sum(1)
            better = cp.asnumpy(g2 > g1)
            if better.any():
                idx = np.flatnonzero(better) * G + ph
                w[idx] = w2[idx]
                s = score(w)
    print(f"round {r + 1}: match {100*s.sum()/total:.2f}%  ({time.time() - t0:.0f} s)")
c = np.corrcoef(np.log(w), np.log(true_im[names[0]]))[0, 1]
print(f"log-correlation of the fitted vector with the true imatrix: {c:.3f}")
