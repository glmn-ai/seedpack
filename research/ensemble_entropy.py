"""How predictable is a seed-based residual? Ensemble experiment.

For each quantised tensor of the chosen files, the seed tensor is dequantised (x) and perturbed K
times within its own rounding uncertainty (Q8_0 seed: +-d/2 per weight; F16 seed: none) and
re-quantised. For every bit of the residual (file XOR prediction from x), c = how many of the K
perturbed predictions flip that bit is a context; an adaptive binary coder conditioned on c (and
on the bit's position in its block) would need sum H(flip | context) bits. Compared with zstd of
the XOR residual.

usage: ensemble_entropy.py QUANT_DIR FILE_PATTERN... [--k K] [--imatrix FILE]
"""
import glob
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import cupy as cp
import numpy as np
import zstandard

from seedpack.gguf_reader import GGUF, TYPE_NAMES, BLOCK, GGML_Q8_0, GGML_F16, GGML_F32, GGML_BF16
from seedpack.repo3 import M29RepoGPU, _keep_base, _read_file, _gguf_tensors
from seedpack.requant import QFUN, load_imatrix
from seedpack.seed import bits_per_weight


def opt(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


K = int(opt("--k", 8))
imf = opt("--imatrix", None)
args = [a for i, a in enumerate(sys.argv[1:], 1) if not a.startswith("--") and not sys.argv[i - 1].startswith("--")]
d, pats = args[0], args[1:]
paths = sorted(glob.glob(os.path.join(d, "*.gguf")))
seed = max(paths, key=lambda p: bits_per_weight(_gguf_tensors(p)))
im = load_imatrix(open(imf, "rb").read()) if imf else {}
m = M29RepoGPU()
m._setup([_keep_base(_read_file(seed))], im)
gq = m.gq
z = zstandard.ZstdCompressor(level=7)


def seed_x(bt):
    """seed tensor -> (float32 values, per-weight half uncertainty) on the GPU"""
    raw = m.base_dev[bt.offset:bt.offset + bt.nbytes]
    n = int(np.prod(bt.shape))
    if bt.type == GGML_Q8_0:
        b = raw.reshape(-1, 34)
        dd = (b[:, 0].astype(cp.uint16) | (b[:, 1].astype(cp.uint16) << 8)).view(cp.float16).astype(cp.float32)
        x = (b[:, 2:].view(cp.int8).astype(cp.float32) * dd[:, None]).ravel()
        h = cp.repeat(cp.abs(dd) / 2, 32)
        return x, h
    if bt.type == GGML_F16:
        return raw.view(cp.float16).astype(cp.float32), cp.zeros(n, cp.float32)
    if bt.type == GGML_BF16:
        return (raw.view(cp.uint16).astype(cp.uint32) << 16).view(cp.float32), cp.zeros(n, cp.float32)
    if bt.type == GGML_F32:
        return raw.view(cp.float32).copy(), cp.zeros(n, cp.float32)
    return gq.dequantize(bt.type, raw, bt.shape[0]), None


rng = cp.random.RandomState(1234)
stats = defaultdict(lambda: {"bits": 0, "zstd": 0, "h_ctx": defaultdict(lambda: np.zeros(2)), "h_pos": defaultdict(lambda: np.zeros(2)), "bytes": 0})
for p in [x for x in paths if any(s in os.path.basename(x) for s in pats) and x != seed]:
    g = GGUF(p, light=True)
    for t in g.tensors:
        if t.type not in QFUN:
            continue
        bt = m.by_name.get(t.name)
        if bt is None:
            continue
        x, h = seed_x(bt)
        if h is None:
            continue
        npr = t.shape[0]
        w = im.get(t.name)
        if w is None:
            w = np.ones(npr, np.float32)
        file_b = np.frombuffer(g.data, np.uint8, t.nbytes, t.offset)
        mid = gq.quantize_tensors(t.type, [(x, npr)], [w])[0]
        res = np.bitwise_xor(file_b, mid)
        flips = np.zeros(t.nbytes * 8, np.uint8)
        for k in range(K):
            u = rng.uniform(-1, 1, x.size, dtype=cp.float32)
            pk = gq.quantize_tensors(t.type, [(x + u * h, npr)], [w])[0]
            flips += np.unpackbits(np.bitwise_xor(pk, mid))
        actual = np.unpackbits(res)
        _, bs = BLOCK[t.type]
        pos = (np.arange(t.nbytes * 8) % (bs * 8)).astype(np.int32)
        s = stats[TYPE_NAMES[t.type]]
        s["bytes"] += t.nbytes
        s["zstd"] += len(z.compress(res.tobytes())) if res.any() else 0
        cnt = np.bincount(flips.astype(np.int64) * 2 + actual, minlength=2 * (K + 1)).reshape(K + 1, 2)
        for c in range(K + 1):
            s["h_ctx"][c] += cnt[c]
        cntp = np.bincount(pos.astype(np.int64) * 2 + actual, minlength=2 * bs * 8).reshape(bs * 8, 2)
        for q in range(bs * 8):
            s["h_pos"][q] += cntp[q]


def H(tab):
    bits = 0.0
    for n0, n1 in tab.values():
        n = n0 + n1
        for v in (n0, n1):
            if v:
                bits -= v * np.log2(v / n)
    return bits / 8


print(f"seed {os.path.basename(seed)}, K={K}, imatrix {'yes' if im else 'no'}")
print(f"{'type':8s} {'MB':>7s} {'zstd MB':>8s} {'H|pos MB':>9s} {'H|ens MB':>9s}  flip rate by ensemble count c=0..K")
for k, s in sorted(stats.items(), key=lambda kv: -kv[1]["zstd"]):
    rates = " ".join(f"{100*v[1]/max(v.sum(),1):.1f}" for _, v in sorted(s["h_ctx"].items()))
    share0 = s["h_ctx"][0].sum() / sum(v.sum() for v in s["h_ctx"].values())
    print(f"{k:8s} {s['bytes']/1e6:7.1f} {s['zstd']/1e6:8.2f} {H(s['h_pos'])/1e6:9.2f} {H(s['h_ctx'])/1e6:9.2f}  "
          f"[{rates}]  (c=0 holds {100*share0:.1f}% of bits)")
