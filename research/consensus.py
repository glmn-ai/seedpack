"""Consensus seeds: does intersecting the value intervals of already-decoded quants shrink QR?

Files are processed from the most to the least bits per weight. For every tensor whose seed is
Q8_0, each weight starts with the seed's interval [x - d/2, x + d/2]; every decoded quant of that
tensor (QR types) intersects it with the interval its code implies. Each file's QR size is
measured with the plain seed and with the refined one (prediction and model both from the
interval midpoint / half-width).

usage: consensus.py QUANT_DIR [--imatrix FILE]
"""
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import cupy as cp
import numpy as np

import qr
from gguf_reader import GGUF, GGML_Q8_0
from repo3 import M29RepoGPU, _keep_base, _read_file, _gguf_tensors
from requant import load_imatrix
from seed import bits_per_weight

d = sys.argv[1]
imf = sys.argv[sys.argv.index("--imatrix") + 1] if "--imatrix" in sys.argv else None
paths = sorted(glob.glob(os.path.join(d, "*.gguf")), key=lambda p: -bits_per_weight(_gguf_tensors(p)))
seed, rest = paths[0], paths[1:]
im = load_imatrix(open(imf, "rb").read()) if imf else {}
m = M29RepoGPU()
m._setup([_keep_base(_read_file(seed))], im)
iv = {}                                            # name -> (lo, hi) for Q8_0-seeded tensors
for n, bt in m.by_name.items():
    if bt.type == GGML_Q8_0:
        iv[n] = qr.intervals(GGML_Q8_0, m.base_dev[bt.offset:bt.offset + bt.nbytes], int(np.prod(bt.shape)))
q8x = {n: m.gq.dequantize(GGML_Q8_0, m.base_dev[m.by_name[n].offset:m.by_name[n].offset + m.by_name[n].nbytes], m.by_name[n].shape[0]) for n in iv}
iv0 = dict(iv)
tot_plain = tot_cons = 0
print(f"seed {os.path.basename(seed)}; imatrix {'yes' if im else 'no'}")
for p in rest:
    g = GGUF(p, light=True)
    ts = [t for t in g.tensors if t.type in qr.QR_TYPES and t.name in iv]
    plain = cons = 0
    by = {}
    for t in ts:
        by.setdefault(t.type, []).append(t)
    for T, tl in by.items():
        w = [im.get(t.name) if t.name in im else np.ones(t.shape[0], np.float32) for t in tl]
        f = [np.frombuffer(g.data, np.uint8, t.nbytes, t.offset) for t in tl]
        bts = [m.by_name[t.name] for t in tl]
        # plain: seed bytes
        pp = m.gq.quantize_tensors(T, [(m._src(bt), t.shape[0]) for t, bt in zip(tl, bts)], w)
        plain += len(qr.encode(T, [(fi, pi, bt.offset, bt.type) for fi, pi, bt in zip(f, pp, bts)], m.base_dev))
        # consensus: midpoint / half-width of the intersected interval (infinite sides: the Q8 bound)
        xs, offs, o = [], [], 0
        for t in tl:
            lo, hi = iv[t.name]
            lo0, hi0 = iv0[t.name]
            nar = (lo != lo0) | (hi != hi0)        # narrowed by a decoded file: midpoint, else the seed value
            x = cp.where(nar, (lo + hi) / 2, q8x[t.name])
            h = cp.where(nar, (hi - lo) / 2, (hi0 - lo0) / 2)
            xs.append(cp.stack([x, h], 1).ravel()); offs.append(o); o += x.size * 8
        buf = cp.concatenate(xs).astype(cp.float32)
        pc = m.gq.quantize_tensors(T, [(x.reshape(-1, 2)[:, 0].copy(), t.shape[0]) for x, t in zip(xs, tl)], w)
        cons += len(qr.encode(T, [(fi, pi, of, qr.EXPLICIT) for fi, pi, of in zip(f, pc, offs)], buf.view(cp.uint8)))
        # refine with this file's codes
        for t, fi in zip(tl, f):
            lo2, hi2 = qr.intervals(T, cp.asarray(fi), int(np.prod(t.shape)))
            lo, hi = iv[t.name]
            nlo, nhi = cp.maximum(lo, lo2), cp.minimum(hi, hi2)
            ok = nlo <= nhi                       # an empty intersection keeps the old interval
            iv[t.name] = (cp.where(ok, nlo, lo), cp.where(ok, nhi, hi))
    tot_plain += plain; tot_cons += cons
    print(f"{os.path.basename(p):30s} QR plain {plain/1e6:7.2f} MB  consensus {cons/1e6:7.2f} MB")
widths = cp.concatenate([(hi - lo)[cp.isfinite(hi - lo)] for lo, hi in iv.values()])
print(f"total QR: plain {tot_plain/1e6:.1f} MB, consensus {tot_cons/1e6:.1f} MB ({100*(1-tot_cons/max(tot_plain,1)):.1f}% less)")
