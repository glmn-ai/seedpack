"""IQ2_XXS / IQ3_XXS residual anatomy: how often do the fields of a predicted block match?

For each 32-weight sub-block: is the 4-bit scale equal; for each 8-weight (IQ2) / 4-weight (IQ3)
group: is the grid index equal, are the sign bits equal; plus the same conditioned on an equal
sub-block scale.

usage: iq_fields.py QUANT_DIR FILE_PATTERN [--imatrix FILE]
"""
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from seedpack.gguf_reader import GGUF
from seedpack.repo3 import M29RepoGPU, _keep_base, _read_file, _gguf_tensors
from seedpack.requant import load_imatrix
from seedpack.seed import bits_per_weight

d, pat = sys.argv[1], sys.argv[2]
imf = sys.argv[sys.argv.index("--imatrix") + 1] if "--imatrix" in sys.argv else None
paths = sorted(glob.glob(os.path.join(d, "*.gguf")))
seed = max(paths, key=lambda p: bits_per_weight(_gguf_tensors(p)))
im = load_imatrix(open(imf, "rb").read()) if imf else {}
m = M29RepoGPU()
m._setup([_keep_base(_read_file(seed))], im)
p = [x for x in paths if pat in os.path.basename(x)][0]
g = GGUF(p, light=True)
acc = {}
for T, name in ((16, "IQ2_XXS"), (18, "IQ3_XXS")):
    ts = [t for t in g.tensors if t.type == T and t.name in m.by_name]
    if not ts:
        continue
    preds = m.gq.quantize_tensors(T, [(m._src(m.by_name[t.name]), t.shape[0]) for t in ts],
                                  [im.get(t.name) if t.name in im else np.ones(t.shape[0], np.float32) for t in ts])
    bs = 66 if T == 16 else 98
    F = np.concatenate([np.frombuffer(g.data, np.uint8, t.nbytes, t.offset) for t in ts]).reshape(-1, bs)
    P = np.concatenate(preds).reshape(-1, bs)
    deq = (F[:, :2] == P[:, :2]).all(1)
    if T == 16:
        q = lambda B: B[:, 2:].reshape(-1, 8, 8)                     # per ib32: 4 grid bytes + u32 (signs, scale)
        fq, pq = q(F), q(P)
        fa = fq[:, :, 4:].copy().view(np.uint32)[..., 0]; pa = pq[:, :, 4:].copy().view(np.uint32)[..., 0]
        sc_eq = (fa >> 28) == (pa >> 28)
        grid_eq = fq[:, :, :4] == pq[:, :, :4]                          # (blocks, 8 sub-blocks, 4 groups)
        sg_eq = np.stack([((fa >> (7 * l)) & 127) == ((pa >> (7 * l)) & 127) for l in range(4)], -1)
    else:
        # IQ3_XXS: d, qs[64] grid bytes (4 weights each), then 8 x u32 (signs 4x7 bits + 4-bit scale)
        fg, pg = F[:, 2:66].reshape(-1, 8, 8), P[:, 2:66].reshape(-1, 8, 8)
        fa = F[:, 66:98].copy().view(np.uint32); pa = P[:, 66:98].copy().view(np.uint32)
        sc_eq = (fa >> 28) == (pa >> 28)
        grid_eq = fg == pg
        sg_eq = np.stack([((fa >> (7 * l)) & 127) == ((pa >> (7 * l)) & 127) for l in range(4)], -1)
    se = sc_eq[..., None]
    print(f"{name}: {len(F)} blocks; d equal {100*deq.mean():.1f}%, sub-scale equal {100*sc_eq.mean():.1f}%, "
          f"grid equal {100*grid_eq.mean():.1f}% (with equal scale {100*grid_eq[np.broadcast_to(se, grid_eq.shape)].mean():.1f}%), "
          f"signs equal {100*sg_eq.mean():.1f}% (with equal scale {100*sg_eq[np.broadcast_to(se, sg_eq.shape)].mean():.1f}%)")
